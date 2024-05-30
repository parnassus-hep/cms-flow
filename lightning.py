import sys
import copy
from typing import Tuple

import torch
import torchdiffeq
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

sys.path.append("./models/")

import matplotlib as mpl
import numpy as np

from models.flow_model import FlowNetV4

from models.sampler import pndm_sampler
from set2setloss import Set2SetLoss
from utils.datasetloader import FastSimDataset

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from custom_scheduler import CosineAnnealingWarmupRestarts


def normalize_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def subplots_with_absolute_sized_axes(
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    axis_width: float,
    axis_height: float,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create axes with exact sizes.

    Spaces axes as far from each other and the figure edges as possible
    within the grid defined by nrows, ncols, and figsize.

    Allows you to share y and x axes, if desired.
    """
    fig = plt.figure(figsize=figsize, **kwargs)
    figwidth, figheight = figsize
    # spacing on each left and right side of the figure
    h_margin = (figwidth - (ncols * axis_width)) / figwidth / ncols / 2
    # spacing on each top and bottom of the figure
    v_margin = (figheight - (nrows * axis_height)) / figheight / nrows / 2
    row_addend = 1 / nrows
    col_addend = 1 / ncols
    inner_ax_width = axis_width / figwidth
    inner_ax_height = axis_height / figheight
    axes = []
    sharex_ax = None
    sharey_ax = None
    for row in range(nrows):
        bottom = (row * row_addend) + v_margin
        for col in range(ncols):
            left = (col * col_addend) + h_margin
            if not axes:
                axes.append(
                    fig.add_axes([left, bottom, inner_ax_width, inner_ax_height])
                )
                if sharex:
                    sharex_ax = axes[0]
                if sharey:
                    sharey_ax = axes[0]
            else:
                axes.append(
                    fig.add_axes(
                        [left, bottom, inner_ax_width, inner_ax_height],
                        sharex=sharex_ax,
                        sharey=sharey_ax,
                    )
                )
    return fig, np.flip(np.asarray(list(axes)).reshape((nrows, ncols)), axis=0)


class FlowLightning(LightningModule):
    def __init__(self, config, comet_logger=None):
        super().__init__()
        torch.manual_seed(1)
        self.config = config

        self.loss = torch.nn.MSELoss()

        self.pred_loss = Set2SetLoss()

        self.sigma = config["sigma"]
        self.n_steps = config["n_steps"]
        self.time_pow = config.get("time_pow", False)
        self.use_global = config.get("use_global", False)
        self.train_npf = config.get("train_npf", False)
        self.use_prev = config.get("use_prev", False)

        self.net = FlowNetV4(config)

        self.FM = TargetConditionalFlowMatcher(sigma=self.sigma)

        self.comet_logger = comet_logger

    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger

    def forward(self, data):
        truth, pflow, mask, scale = data
        x0 = torch.randn_like(pflow)
        if self.time_pow:
            t = torch.Tensor(np.random.power(3, size=pflow.shape[0])).to(pflow.device)
        else:
            t = None
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, pflow, t=t)
        pf_mask = mask[..., 1]
        if self.use_prev:
            out_prev = torch.zeros_like(ut, device=ut.device)
            if torch.rand(1) > 0.5:
                out_prev = self.net(
                    xt,
                    truth,
                    mask,
                    timestep=t,
                    scale=scale if self.use_global else None,
                    fs_data_prev=out_prev,
                ).detach()
            out = self.net(
                xt,
                truth,
                mask,
                timestep=t,
                scale=scale if self.use_global else None,
                fs_data_prev=out_prev,
                return_npf=True,
            )
        else:
            out = self.net(
                xt,
                truth,
                mask,
                timestep=t,
                scale=scale if self.use_global else None,
                return_npf=True,
            )
        if self.train_npf:
            vt, loss_lik = out
            loss_lik = torch.mean(loss_lik) * 0.1
            if loss_lik.isnan():
                loss_lik = 0
        else:
            vt = out
            loss_lik = 0
        if pf_mask.sum() == 0:
            loss = 0
        else:
            loss = self.loss(ut[pf_mask], vt[pf_mask])

        return loss, loss_lik

    @torch.no_grad()
    def sample(
        self,
        truth,
        pflow,
        mask,
        n_steps=None,
        method="dopri5",
        scale=None,
        save_seq=False,
    ):
        if n_steps is None:
            n_steps = self.n_steps

        if method in ["pndm"]:
            return pndm_sampler(
                self.net,
                truth,
                pflow,
                mask,
                scale,
                n_steps=n_steps,
                save_seq=save_seq,
            )[0]
        else:
            return torchdiffeq.odeint(
                lambda t, x: self.net(
                    x,
                    truth,
                    mask=mask,
                    timestep=t * torch.ones(x.shape[0], device=x.device),
                    scale=scale,
                ),
                torch.randn_like(pflow, device=truth.device),
                torch.linspace(0, 1, n_steps, device=truth.device),
                method=method,
                atol=1e-4,
                rtol=1e-4,
            )[-1, ...]

    def training_step(self, data, batch_idx):
        loss, loss_lik = self.forward(data)

        self.log(
            "train_loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )
        self.log(
            "train_loss_npf",
            loss_lik,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )

        return loss + loss_lik

    def validation_step(self, data, batch_idx):
        loss, loss_lik = self.forward(data)
        truth, pflow, mask, scale = data
        return_dict = {}
        return_dict["val_loss"] = loss
        return_dict["val_loss_npf"] = loss_lik
        with torch.no_grad():
            pred = self.sample(
                truth,
                pflow,
                mask,
                scale=scale if self.use_global else None,
                method="pndm",
            )
            fs_mask = None
            if self.train_npf:
                n_pf = self.net.sample_npf(truth, mask)
                npf_target = mask[..., 1].sum(-1).cpu()
                n_pf = n_pf.round().int()
                fs_mask = torch.zeros_like(mask)[..., 0]
                for i in range(fs_mask.shape[0]):
                    fs_mask[i, : n_pf[i]] = 1
                fs_mask = fs_mask.bool()
                self.log(
                    "npf_loss",
                    (n_pf - npf_target).float().abs().mean(),
                    batch_size=data[0].shape[0],
                    sync_dist=True,
                )
            pred_loss = self.get_pred_loss(pred, pflow, mask=mask)
        self.log(
            "val_loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )
        self.log(
            "val_loss_npf",
            loss_lik,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )
        for key, val in pred_loss.items():
            self.log(
                f"pred_{key}",
                val,
                batch_size=data[0].shape[0],
                sync_dist=True,
            )
            return_dict[f"pred_{key}"] = val

        return_dict["truth"] = truth.cpu()
        return_dict["pflow"] = pflow.cpu()
        return_dict["mask"] = mask.cpu()
        return_dict["scale"] = scale.cpu()
        return_dict["fs"] = pred.cpu()
        return_dict["fs_mask"] = fs_mask.cpu()

        return return_dict

    def training_epoch_end(self, outputs):
        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)
            self.lr_schedulers().step()

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_pred_loss = torch.stack([x["pred_total_loss"] for x in outputs]).mean()
        self.log("val_loss_avg", avg_loss, sync_dist=True)
        self.log("pred_loss_avg", avg_pred_loss, sync_dist=True)
        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)

        truth = torch.cat([x["truth"] for x in outputs], dim=0)
        pflow = torch.cat([x["pflow"] for x in outputs], dim=0)
        mask = torch.cat([x["mask"] for x in outputs], dim=0)
        scale = torch.cat([x["scale"] for x in outputs], dim=0)
        fs = torch.cat([x["fs"] for x in outputs], dim=0)
        fs_mask = torch.cat([x["fs_mask"] for x in outputs], dim=0)

        val_jet_pt_mse = self.log_image(truth, pflow, mask, scale, fs, fs_mask)
        self.log("val_jet_pt_mse", val_jet_pt_mse, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learningrate"])
        if self.config["lr_scheduler"] is False:
            return optimizer
        else:
            print("\nscheduler exists!\n")
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=400,
                warmup_steps=25,
                max_lr=self.config["learningrate"],
                min_lr=1e-5,
                gamma=0.8,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        if "reduce_ds_train" in self.config:
            reduce_ds = self.config["reduce_ds_train"]
        else:
            reduce_ds = 1

        dataset = FastSimDataset(
            self.config["truth_path_train"],
            self.config,
            reduce_ds=reduce_ds,
            entry_start=self.config["entry_start_train"],
        )
        loader = DataLoader(
            dataset,
            num_workers=self.config["num_workers"],
            batch_size=self.config["batchsize"],
            drop_last=True,
            shuffle=True,
            pin_memory=False,
            prefetch_factor=4,
        )
        return loader

    def val_dataloader(self):
        if "reduce_ds_valid" in self.config:
            reduce_ds = self.config["reduce_ds_valid"]
        else:
            reduce_ds = 1
        dataset = FastSimDataset(
            self.config["truth_path_valid"],
            self.config,
            reduce_ds=reduce_ds,
            entry_start=self.config["entry_start_valid"],
        )

        loader = DataLoader(
            dataset,
            num_workers=0,
            batch_size=self.config.get("val_batchsize", self.config["batchsize"]),
            drop_last=True,
            pin_memory=False,
            shuffle=False,
        )
        return loader

    def get_pred_loss(self, pred, target, mask):
        loss = {}
        loss = self.pred_loss(pred, target, mask)
        return loss

    def log_image(self, truth, pflow, mask, scale, fs, fs_mask=None):
        # fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=200)#, tight_layout=True)
        fig, axs = subplots_with_absolute_sized_axes(
            2, 3, figsize=(16, 10), axis_width=4, axis_height=4, dpi=200
        )
        canvas = FigureCanvas(fig)

        truth_mask = mask[..., 0].cpu().numpy()
        pflow_mask = mask[..., 1].cpu().numpy()
        if fs_mask is None:
            fs_mask = pflow_mask
        else:
            fs_mask = fs_mask.cpu().bool().numpy()
        # pt eta phi plots
        pt_mean = scale[..., 0].view(-1, 1).cpu().numpy()
        pt_std = scale[..., 1].view(-1, 1).cpu().numpy()
        eta_mean = scale[..., 2].view(-1, 1).cpu().numpy()
        eta_std = scale[..., 3].view(-1, 1).cpu().numpy()
        phi_mean = scale[..., 4].view(-1, 1).cpu().numpy()
        phi_std = scale[..., 5].view(-1, 1).cpu().numpy()

        truth_pt = truth[..., 0].cpu().numpy() * pt_std + pt_mean

        pflow_pt = pflow[..., 0].cpu().numpy() * pt_std + pt_mean
        pflow_eta = pflow[..., 1].cpu().numpy() * eta_std + eta_mean
        pflow_phi = normalize_phi(pflow[..., 2].cpu().numpy() * phi_std + phi_mean)

        fs_pt = fs[..., 0].cpu().numpy() * pt_std + pt_mean
        fs_eta = fs[..., 1].cpu().numpy() * eta_std + eta_mean
        fs_phi = normalize_phi(fs[..., 2].cpu().numpy() * phi_std + phi_mean)

        scatter_mask = fs_mask & pflow_mask
        cmap = copy.copy(plt.get_cmap("PuRd"))
        cmap.set_under("white")

        def _add_hist2d(fig, ax, pf_data, fs_data, bins):
            h, xe, ye, im = ax.hist2d(
                pf_data,
                fs_data,
                bins=bins,
                cmap=cmap,
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        bins = np.linspace(6, 14, 100)
        _add_hist2d(fig, axs[0][0], pflow_pt[scatter_mask], fs_pt[scatter_mask], bins)
        axs[0][0].set_title("log(pT)")
        axs[0][0].set_xlabel("PFlow")
        axs[0][0].set_ylabel("FastSim")
        bins = np.linspace(-3, 3, 100)
        _add_hist2d(fig, axs[0][1], pflow_eta[scatter_mask], fs_eta[scatter_mask], bins)
        axs[0][1].set_title("eta")
        axs[0][1].set_xlabel("PFlow")
        axs[0][1].set_ylabel("FastSim")

        bins = np.linspace(-np.pi, np.pi, 100)
        _add_hist2d(fig, axs[0][2], pflow_phi[scatter_mask], fs_phi[scatter_mask], bins)
        axs[0][2].set_title("phi")
        axs[0][2].set_xlabel("PFlow")
        axs[0][2].set_ylabel("FastSim")

        truth_jet_pt = (np.exp(truth_pt * truth_mask) * truth_mask).sum(
            -1
        ).flatten() / 1000
        pflow_jet_pt = (np.exp(pflow_pt * pflow_mask) * pflow_mask).sum(
            -1
        ).flatten() / 1000
        fs_jet_pt = (np.exp(fs_pt * fs_mask) * fs_mask).sum(-1).flatten() / 1000

        hadron_pt = self.config["truth_path_train"].split("/")[-2][3:]
        if "_" in hadron_pt:
            bins = np.linspace(
                int(hadron_pt.split("_")[0]) - 200,
                int(hadron_pt.split("_")[1]) + 300,
                200,
            )
        else:
            try:
                bins = np.linspace(
                    max(int(hadron_pt) - 200, 0), int(hadron_pt) + 300, 200
                )
            except:
                bins = np.linspace(0, 140, 100)

        def _add_to_hist(ax, data, bins, label):
            ax.hist(
                data,
                bins=bins,
                histtype="step",
                label=f"{label} {np.mean(data):.2f}/{np.std(data):.2f}",
            )

        _add_to_hist(axs[1][0], truth_jet_pt, bins, "Truth")
        _add_to_hist(axs[1][0], pflow_jet_pt, bins, "PFlow")
        _add_to_hist(axs[1][0], fs_jet_pt, bins, "FastSim")
        axs[1][0].set_title("Jet pT")
        axs[1][0].legend()

        bins = np.linspace(-200, 200, 100)
        _add_to_hist(axs[1][1], pflow_jet_pt - truth_jet_pt, bins, "PFlow")
        _add_to_hist(axs[1][1], fs_jet_pt - truth_jet_pt, bins, "FastSim")
        axs[1][1].set_title("Jet pT Residual")
        axs[1][1].legend()

        bins = np.linspace(-30.5, 30.5, 62)
        n_truth = truth_mask.sum(-1).flatten()
        n_pflow = pflow_mask.sum(-1).flatten()
        n_fs = fs_mask.sum(-1).flatten()
        _add_to_hist(axs[1][2], n_pflow - n_truth, bins, "PFlow")
        _add_to_hist(axs[1][2], n_fs - n_truth, bins, "FastSim")
        axs[1][2].set_title("N PF Residual")
        axs[1][2].legend()

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )
        for i in range(2):
            for j in range(3):
                if axs[i, j] is not None:
                    axs[i, j].set_box_aspect(1)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        if self.logger is not None:
            self.logger.experiment.log_image(
                image_data=image,
                name=f"ED_{self.current_epoch}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"ED_{self.current_epoch}.png")
        plt.close(fig)
        return np.mean((fs_jet_pt - pflow_jet_pt) ** 2)
