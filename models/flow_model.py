import torch
import torch.nn as nn
from modules import DenseNetwork, SineCosineEncoding, TimestepEmbedder
from transformer import MultiHeadedAttentionBlock, TransformerEncoder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CALayer(nn.Module):
    def __init__(
        self, hidden_dim, num_heads, mha_config=None, mlp_ratio=4.0, do_gru=False
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_k = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = DenseNetwork(
            inpt_dim=hidden_dim,
            outp_dim=hidden_dim,
            hddn_dim=[mlp_hidden_dim],
            act_h="gelu",
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim * 6, bias=True)
        )
        if mha_config is None:
            mha_config = {}
        self.attn = MultiHeadedAttentionBlock(
            model_dim=hidden_dim,
            do_selfattn=False,
            do_gru=False,
            num_heads=num_heads,
            new_mask=True,
            **mha_config,
        )

    def forward(self, seq_q, seq_k, ctxt, mask_q=None, mask_k=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(ctxt).chunk(6, dim=-1)
        seq_q = seq_q + gate_msa.unsqueeze(1) * self.attn(
            q=modulate(self.norm1(seq_q), shift_msa, scale_msa),
            k=self.norm_k(seq_k),
            kv_mask=mask_k,
            q_mask=mask_q,
        )
        seq_q = seq_q + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(seq_q), shift_mlp, scale_mlp)
        )
        return seq_q


class CABlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mha_config=None, mlp_ratio=4.0):
        super().__init__()

        self.layer_a = CALayer(hidden_dim, num_heads, mha_config, mlp_ratio)
        self.layer_b = CALayer(hidden_dim, num_heads, mha_config, mlp_ratio)

    def forward(self, seq_a, seq_b, ctxt, mask_a=None, mask_b=None):
        seq_b = self.layer_b(
            seq_q=seq_b, seq_k=seq_a, ctxt=ctxt, mask_q=mask_b, mask_k=mask_a
        )
        seq_a = self.layer_a(
            seq_q=seq_a, seq_k=seq_b, ctxt=ctxt, mask_q=mask_a, mask_k=mask_b
        )

        return seq_a, seq_b


class FlowNetV4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config["hidden_dim"]
        num_heads = config["num_heads"]
        num_dit_layers = config["num_dit_layers"]

        act = config["act"]
        self.hidden_dim = hidden_dim
        self.act = act

        self.use_pos_embd = config["use_pos_embd"]
        self.use_global = config.get("use_global", False)
        self.train_npf = config.get("train_npf", False)
        self.use_prev = config.get("use_prev", False)
        self.masked_mean = config.get("masked_mean", False)

        self.time_embedding = TimestepEmbedder(hidden_dim)

        if self.use_pos_embd:
            self.pos_embedding = SineCosineEncoding(
                hidden_dim // 4, min_value=0, max_value=256
            )

        self.truth_init = DenseNetwork(
            inpt_dim=4 + (hidden_dim // 4 if self.use_pos_embd else 0),
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=act,
        )
        fs_data_dim = 3
        self.fs_init = DenseNetwork(
            inpt_dim=fs_data_dim
            + (hidden_dim // 4 if self.use_pos_embd else 0)
            + fs_data_dim * self.use_prev,
            outp_dim=hidden_dim,
            hddn_dim=[hidden_dim, hidden_dim],
            act_h=act,
        )
        if self.use_global:
            self.global_embedding = DenseNetwork(
                inpt_dim=6,
                outp_dim=hidden_dim,
                hddn_dim=[hidden_dim, hidden_dim],
                act_h=act,
            )

        self.truth_embedding = TransformerEncoder(
            model_dim=hidden_dim,
            num_layers=2,
            mha_config={
                "num_heads": num_heads,
                "new_mask": True,
            },
            dense_config={
                "act_h": act,
                "hddn_dim": 2 * hidden_dim,
            },
        )

        self.ca_blocks = nn.ModuleList(
            [CABlock(hidden_dim, num_heads) for _ in range(num_dit_layers)]
        )

        self.final_layer = DenseNetwork(
            inpt_dim=hidden_dim,
            outp_dim=fs_data_dim,
            hddn_dim=[hidden_dim, hidden_dim, hidden_dim],
            act_h=act,
            ctxt_dim=2 * hidden_dim,
        )
        idx = torch.arange(config["max_particles"]).unsqueeze(0)
        self.register_buffer("idx", idx)

        self.initialize_weights()

        if self.train_npf:
            self.npf_model = DenseNetwork(
                inpt_dim=hidden_dim + 1,
                outp_dim=config["max_particles"],
                hddn_dim=[hidden_dim // 2, hidden_dim // 2],
                act_h=act,
            )
            self.npf_loss = nn.CrossEntropyLoss(reduction="none")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ca_blocks:
            nn.init.constant_(block.layer_a.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.layer_a.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.layer_b.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.layer_b.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        fs_data,
        truth_data,
        mask,
        timestep,
        scale=None,
        fs_data_prev=None,
        return_npf=False,
    ):
        truth_mask = mask[..., 0]
        pf_mask = mask[..., 1]

        if self.use_pos_embd:
            idx = self.idx.expand(fs_data.size(0), -1)
            pos_embd = self.pos_embedding(idx)[:, : fs_data.size(1)]
            fs_data_ = torch.cat([fs_data, pos_embd], dim=-1)
            truth_data_ = torch.cat([truth_data, pos_embd], dim=-1)
        else:
            fs_data = fs_data_
            truth_data_ = truth_data
        truth_embd = self.truth_init(truth_data_)
        truth_embd = self.truth_embedding(truth_embd, mask=truth_mask)

        if self.masked_mean:
            truth_ctxt = torch.sum(
                truth_embd * truth_mask.unsqueeze(-1), dim=1
            ) / torch.sum(truth_mask, dim=1, keepdim=True)
        else:
            truth_ctxt = torch.mean(
                truth_embd, dim=1
            )  # Average over the sequence length
        time_embd = self.time_embedding(timestep)

        ctxt = torch.cat(
            [
                truth_ctxt + self.global_embedding(scale)
                if self.use_global
                else time_embd,
                time_embd,
            ],
            -1,
        )
        if fs_data_prev is not None and self.use_prev:
            fs_data_ = torch.cat([fs_data_, fs_data_prev], -1)
        fs_embd = self.fs_init(fs_data_)
        for block in self.ca_blocks:
            truth_embd, fs_embd = block(
                truth_embd, fs_embd, ctxt, mask_a=truth_mask, mask_b=pf_mask
            )
        fs_out = self.final_layer(fs_embd, ctxt)
        if self.train_npf and return_npf:
            num_pf = mask[..., 1].sum(-1).long()
            num_tr = mask[..., 0].sum(-1).float().view(-1, 1)
            npf_logits = self.npf_model(torch.cat([truth_ctxt, num_tr], dim=-1))
            npf_loss = self.npf_loss(npf_logits, num_pf)
            return fs_out, npf_loss
        return fs_out

    def get_embd(self, truth_data, truth_mask):
        if self.use_pos_embd:
            idx = self.idx.expand(truth_data.size(0), -1)
            pos_embd = self.pos_embedding(idx)[:, : truth_data.size(1)]
            truth_data_ = torch.cat([truth_data, pos_embd], dim=-1)
        else:
            truth_data_ = truth_data
        truth_embd = self.truth_init(truth_data_)
        truth_embd = self.truth_embedding(truth_embd, mask=truth_mask)
        if self.masked_mean:
            truth_ctxt = torch.sum(
                truth_embd * truth_mask.unsqueeze(-1), dim=1
            ) / torch.sum(truth_mask, dim=1, keepdim=True)
        else:
            truth_ctxt = torch.mean(
                truth_embd, dim=1
            )  # Average over the sequence length
        return truth_ctxt

    @torch.no_grad()
    def sample_npf(self, truth_data, mask):
        truth_mask = mask[..., 0]
        truth_ctxt = self.get_embd(truth_data, truth_mask)
        num_tr = mask[..., 0].sum(-1).float().view(-1, 1)
        npf_logits = self.npf_model(torch.cat([truth_ctxt, num_tr], dim=-1))
        pred_num_pf = torch.multinomial(
            npf_logits.softmax(-1), 1, replacement=True
        ).squeeze(1)
        return pred_num_pf.cpu()
