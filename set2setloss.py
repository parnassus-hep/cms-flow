import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class Set2SetLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.regression_loss = nn.MSELoss(reduction="none")

    def forward(self, input, target, mask):
        ### new code !!!
        bs = len(input)
        pf_mask = mask[..., 1].cpu()
        new_mask = pf_mask.unsqueeze(1).expand(-1, target.size(1), -1)

        new_input = input.cpu().unsqueeze(1).expand(-1, target.size(1), -1, -1)
        new_target = target.cpu().unsqueeze(2).expand(-1, -1, input.size(1), -1)

        pdist = self.regression_loss(new_input, new_target)

        pdist_pt = pdist[..., 0]
        pdist_eta = pdist[..., 1]
        pdist_phi = pdist[..., 2]

        pdist = pdist.mean(-1)
        pdist = pdist * new_mask
        pdist_pt = pdist_pt * new_mask
        pdist_eta = pdist_eta * new_mask
        pdist_phi = pdist_phi * new_mask

        pdist_ = pdist.detach().numpy()
        indices = np.array(
            [linear_sum_assignment(p) for p in pdist_]
        )  # indices shape (b,2,N)

        spicy_mat = torch.zeros((bs), device=pdist.device)
        spicy_mat_pt = torch.zeros((bs), device=pdist.device)
        spicy_mat_eta = torch.zeros((bs), device=pdist.device)
        spicy_mat_phi = torch.zeros((bs), device=pdist.device)
        for idx_i in range(bs):
            indices_i = indices.shape[2] * indices[idx_i, 0] + indices[idx_i, 1]

            losses = torch.gather(
                pdist[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss = losses.mean(0)

            spicy_mat[idx_i] = total_loss

            losses_pt = torch.gather(
                pdist_pt[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss_pt = losses_pt.mean(0)
            spicy_mat_pt[idx_i] = total_loss_pt

            losses_eta = torch.gather(
                pdist_eta[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss_eta = losses_eta.mean(0)
            spicy_mat_eta[idx_i] = total_loss_eta

            losses_phi = torch.gather(
                pdist_phi[idx_i].flatten(0, 1),
                0,
                torch.from_numpy(indices_i).to(device=pdist.device),
            )
            total_loss_phi = losses_phi.mean(0)
            spicy_mat_phi[idx_i] = total_loss_phi

        return {
            "total_loss": spicy_mat.mean().to(input.device),
            "pt_loss": spicy_mat_pt.mean(),
            "eta_loss": spicy_mat_eta.mean(),
            "phi_loss": spicy_mat_phi.mean(),
        }
