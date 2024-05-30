"""
Some classes to describe transformer architectures.
https://github.com/rodem-hep/nu2flows
"""

import math
from typing import Mapping, Optional, Union

import torch as T
import torch.nn as nn
from modules import DenseNetwork
from torch.nn.functional import dropout, softmax


def merge_kqv_masks(
    q_mask: Union[T.BoolTensor, None],
    kv_mask: Union[T.BoolTensor, None],
    attn_mask: Union[T.BoolTensor, None],
    q_shape: T.Size,
    k_shape: T.Size,
    device: T.device,
) -> T.BoolTensor:
    """Create a full attention mask which incoporates the padding information.

    Using pytorch transformer convention:
        False: Real node
        True:  Zero padded
    """
    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = T.full(q_shape[:-1], False, device=device)
        if kv_mask is None:
            kv_mask = T.full(k_shape[:-1], False, device=device)
        merged_mask = ~q_mask.unsqueeze(-1) | ~kv_mask.unsqueeze(-2)

    # If attention mask exists then it must be included
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask | merged_mask

    if merged_mask is not None:
        merged_mask = merged_mask.unsqueeze(1)

    return ~merged_mask


def merge_masks(
    # q_mask: Union[T.BoolTensor, None],
    kv_mask: Union[T.BoolTensor, None],
    attn_mask: Union[T.BoolTensor, None],
    attn_bias: Union[T.Tensor, None],
    query: T.Tensor,
) -> Union[None, T.BoolTensor]:
    """Create a full attention mask which incoporates the padding information
    and the bias terms.

    New philosophy is just to define a kv_mask, and let the q_mask be
    ones. Let the padded nodes receive what they want! Their outputs
    dont matter and they don't add to computation anyway!!!
    """

    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    if kv_mask is not None:
        merged_mask = kv_mask.unsqueeze(-2).expand(-1, query.shape[-2], -1)

    # If ontop of that we defined a custom attention mask then that is added
    if attn_mask is not None:
        merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    # Unsqueeze the mask to give it a dimension for num_head broadcasting
    if merged_mask is not None:
        merged_mask = merged_mask.unsqueeze(1)

    # If the attention bias exists, convert to a float and add to the mask
    if attn_bias is not None:
        if merged_mask is not None:
            merged_mask = T.where(merged_mask, 0, -T.inf).type(query.dtype)
            merged_mask = merged_mask + attn_bias.permute(0, 3, 1, 2)
        else:
            merged_mask = attn_bias.permute(0, 3, 1, 2)

    return merged_mask


def scaled_dot_product_attention(
    query: T.Tensor,
    key: T.Tensor,
    value: T.Tensor,
    attn_mask: Optional[T.BoolTensor] = None,
    attn_bias: Optional[T.Tensor] = None,
    dropout_p: float = 0.0,
) -> T.Tensor:
    """DEPRECATED! THE PYTORCH-2.0 IMPLEMENATION IS 25% FASTER AND HAS A
    REDUCED MEMORY OVERHEAD SO MY ATTENTION LAYERS HAVE SWITCHED OVER TO
    THAT!!!

    Apply the attention using the scaled dot product between the key query
    and key tensors, then matrix multiplied by the value.

    Note that the attention scores are ordered in recv x send, which is the opposite
    to how I usually do it for the graph network, which is send x recv

    We use masked fill -T.inf as this kills the padded key/values elements but
    introduces nans for padded query elements. We could used a very small number like
    -1e9 but this would need to scale with if we are using half precision.

    Args:
        query: Batched query sequence of tensors (b, h, s, f)
        key: Batched key sequence of tensors (b, h, s, f)
        value: Batched value sequence of tensors (b, h, s, f)
        attn_mask: The attention mask, used to blind certain combinations of k,q pairs
        attn_bias: Extra weights to combine with attention weights
        drp: Dropout probability
    """
    DeprecationWarning("Dont use this! Switch to pytorch 2.0 built in version!")

    # Perform the matrix multiplication
    scores = query @ key.transpose(-2, -1) / math.sqrt(key.shape[-1])

    # Add the bias terms if present
    if attn_bias is not None:  # Move the head dimension to the first
        scores = scores + attn_bias

    # Mask away the scores between invalid elements in sequence
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, -T.inf)

    # Apply the softmax function per head feature
    scores = softmax(scores, dim=-1)

    # Kill the nans introduced by the padded query elements
    if attn_mask is not None:
        scores = T.nan_to_num(scores)

    # Apply dropout to the attention scores
    scores = dropout(scores, p=dropout_p)

    # Finally multiply these scores by the output
    scores = scores @ value

    return scores


class MultiHeadedAttentionBlock(nn.Module):
    """Generic Multiheaded Attention.

    Takes in three sequences with dim: (batch, sqeuence, features)
    - q: The primary sequence queries (determines output sequence length)
    - k: The attending sequence keys (determines incoming information)
    - v: The attending sequence values

    In a message passing sense you can think of q as your receiver nodes, v and k
    are the information coming from the sender nodes.

    When q == k(and v) this is a SELF attention operation
    When q != k(and v) this is a CROSS attention operation

    ===

    Block operations:

    1) Uses three linear layers to project the sequences.
    - q = q_linear * q
    - k = k_linear * k
    - v = v_linear * v

    2) Outputs are reshaped to add a head dimension, and transposed for matmul.
    - features = model_dim = head_dim * num_heads
    - dim becomes: batch, num_heads, sequence, head_dim

    3) Passes these through to the attention module (message passing)
    - In standard transformers this is the scaled dot product attention
    - Also takes additional dropout param to mask the attention

    4) Flatten out the head dimension and pass through final linear layer
    - Optional layer norm before linear layer using `do_layer_norm=True`
    - The output can also be zeroed on init using `init_zeros=True`
    - results are same as if attention was done seperately for each head and concat
    - dim: batch, q_seq, head_dim * num_heads
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 1,
        drp: float = 0,
        init_zeros: bool = False,
        do_selfattn: bool = False,
        do_layer_norm: bool = False,
        do_gru: bool = False,
        new_mask: bool = False,
    ) -> None:
        """
        Args:
            model_dim: The dimension of the model
            num_heads: The number of different attention heads to process in parallel
                - Must allow interger division into model_dim
            drp: The dropout probability used in the MHA operation
            init_zeros: If the final linear layer is initialised with zero weights
            do_selfattn: Only self attention should only be used if the
                q, k, v are the same, this allows slightly faster matrix multiplication
                at the beginning
            do_layer_norm: If a layernorm is applied before the output final linear
                projection (Only really needed with deep models)
        """
        super().__init__()

        # Define model base attributes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.do_selfattn = do_selfattn
        self.drp = drp
        self.do_layer_norm = do_layer_norm
        self.do_gru = do_gru
        self.new_mask = new_mask

        # Check that the dimension of each head makes internal sense
        if self.head_dim * num_heads != model_dim:
            raise ValueError("Model dimension must be divisible by number of heads!")

        # Initialise the weight matrices (only 1 for do self attention)
        if do_selfattn:
            self.all_linear = nn.Linear(model_dim, 3 * model_dim, bias=False)
        else:
            self.q_linear = nn.Linear(model_dim, model_dim, bias=False)
            self.k_linear = nn.Linear(model_dim, model_dim, bias=False)
            self.v_linear = nn.Linear(model_dim, model_dim, bias=False)

        # The optional (but advised) layer normalisation
        if do_layer_norm:
            self.layer_norm = nn.LayerNorm(model_dim)

        # Set the output linear layer weights and bias terms to zero
        self.out_linear = nn.Linear(model_dim, model_dim)
        if init_zeros:
            self.out_linear.weight.data.fill_(0)
            self.out_linear.bias.data.fill_(0)

    def forward(
        self,
        q: T.Tensor,
        k: Optional[T.Tensor] = None,
        v: Optional[T.Tensor] = None,
        kv_mask: Optional[T.BoolTensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        q_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        """
        Args:
            q: The main sequence queries (determines the output length)
            k: The incoming information keys
            v: The incoming information values
            q_mask: Shows which elements of the main sequence are real
            kv_mask: Shows which elements of the attn sequence are real
            attn_mask: Extra mask for the attention matrix (eg: look ahead)
            attn_bias: Extra bias term for the attention matrix (eg: edge features)
        """

        # Store the batch size, useful for reshaping
        b_size, seq, feat = q.shape

        # If only q and q_mask are provided then we automatically apply self attention
        if k is None:
            k = q
        if v is None:
            v = k
        # if q_mask is None:
        #     q_mask = kv_mask
        # Work out the masking situation, with padding, no peaking etc
        if self.new_mask:
            merged_mask = merge_kqv_masks(
                q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device
            )
        else:
            merged_mask = merge_masks(kv_mask, attn_mask, attn_bias, q)

        # Generate the q, k, v projections
        if self.do_selfattn:
            q_out, k_out, v_out = self.all_linear(q).chunk(3, -1)
        else:
            q_out = self.q_linear(q)
            k_out = self.k_linear(k)
            v_out = self.v_linear(v)

        # Break final dim, transpose to get dimensions: B,H,Seq,Hdim
        shape = (b_size, -1, self.num_heads, self.head_dim)
        q_out = q_out.view(shape).transpose(1, 2)
        k_out = k_out.view(shape).transpose(1, 2)
        v_out = v_out.view(shape).transpose(1, 2)

        # Calculate the new sequence values
        a_out = scaled_dot_product_attention(
            q_out,
            k_out,
            v_out,
            attn_mask=merged_mask,
            dropout_p=self.drp if self.training else 0,
        )

        # Concatenate the all of the heads together to get shape: B,Seq,F
        a_out = a_out.transpose(1, 2).contiguous().view(b_size, -1, self.model_dim)

        # Pass through the optional normalisation layer
        if self.do_layer_norm:
            a_out = self.layer_norm(a_out)

        # Pass through final linear layer
        return self.out_linear(a_out)


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer based on the GPT-2+Normformer style
    arcitecture.

    We choose a cross between Normformer and FoundationTransformers as they have often
    proved to be the most stable to train
    https://arxiv.org/abs/2210.06423
    https://arxiv.org/abs/2110.09456

    It contains:
    - Multihead(self)Attention block
    - A dense network

    Layernorm is applied before each operation
    Residual connections are used to bypass each operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: The embedding dimension of the transformer block
            mha_config: Keyword arguments for multiheaded-attention block
            dense_config: Keyword arguments for feed forward network
            ctxt_dim: Context dimension,
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_config = dense_config or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.self_attn = MultiHeadedAttentionBlock(
            model_dim, do_selfattn=True, **mha_config
        )
        self.dense = DenseNetwork(
            model_dim, outp_dim=model_dim, ctxt_dim=ctxt_dim, **dense_config
        )

        # The pre MHA and pre FFN layer normalisations
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Optional[T.Tensor] = None,
        attn_bias: Optional[T.Tensor] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        "Pass through the layer using residual connections and layer normalisation"
        x = x + self.self_attn(
            self.norm1(x),
            kv_mask=mask,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
            q_mask=mask,
        )
        x = x + self.dense(self.norm2(x), ctxt)
        return x


class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final
    normalisation step.

    Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 3,
        mha_config: Optional[Mapping] = None,
        dense_config: Optional[Mapping] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_config: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_config, dense_config, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_norm(x)
