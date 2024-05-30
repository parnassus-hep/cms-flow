import argparse
import json
import os
import re

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightning import FlowLightning
from utils.datasetloader import FastSimDataset


def reshape_phi(phi):
    return torch.arctan2(torch.sin(phi), torch.cos(phi))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("-p", "--checkpoint", type=str, required=True)
parser.add_argument("-n", "--n_steps", type=int, default=25)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-e", "--eval_dir", type=str, default="evals")
parser.add_argument("-ne", "--num_events", type=int, default=100_00)
parser.add_argument("-bs", "--batch_size", type=int, default=200)
parser.add_argument("-npf", action="store_true")
parser.add_argument("--test_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default="")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

with open(args.config, "r") as fp:
    config = json.load(fp)

ckpt_path = args.checkpoint
epoch = re.search(r"(?<=epoch=).\d*", ckpt_path)
if epoch is None:
    epoch = "last"
else:
    epoch = epoch.group(0)

net = FlowLightning(config)
if os.path.exists(f"{args.eval_dir}") is False:
    os.makedirs(f"{args.eval_dir}")
eval_path = f"{args.eval_dir}/{args.prefix}{config['name']}_{epoch}_{args.n_steps}_pndm_{'npf' if args.npf else ''}.root"

checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
net.load_state_dict(checkpoint["state_dict"])

torch.set_grad_enabled(False)
net.eval()
net.net.eval()
device = torch.device("cuda")
net.to(device)

test_path = config["truth_path_test"] if args.test_path is None else args.test_path
with uproot.open(test_path) as f:
    if "jet_tree" in f:
        t = f["jet_tree"]
        jet_data = {
            key: t[key].array(library="np", entry_stop=args.num_events).flatten()
            for key in t.keys()
        }
dataset = FastSimDataset(
    test_path,
    config,
    reduce_ds=args.num_events,
    entry_start=0,
)

loader = DataLoader(
    dataset,
    num_workers=4,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=False,
)

n_events = len(dataset)
l_pt_tr, l_pt_pf, l_pt_fs = (
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
)
l_eta_tr, l_eta_pf, l_eta_fs = (
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
)
l_phi_tr, l_phi_pf, l_phi_fs = (
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
)
l_ind_tr, l_ind_pf, l_ind_fs = (
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
)
l_class_tr, l_class_pf, l_class_fs = (
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
    np.zeros((n_events, dataset.max_particles)),
)
n = 0
for i, (truth, pflow, mask, scale) in tqdm(enumerate(loader), total=len(loader)):
    fs_mask = None
    sample_mask = mask
    if net.train_npf and args.npf:
        n_pf_pred = net.net.sample_npf(truth.to(device), mask.to(device))
        fs_mask = torch.zeros_like(mask)[..., 1]
        for i in range(fs_mask.shape[0]):
            fs_mask[i, : n_pf_pred[i]] = 1
        fs_mask = fs_mask.bool()
        sample_mask = torch.stack([mask[..., 0], fs_mask], -1)

    fs = net.sample(
        truth.to(device),
        pflow.to(device),
        sample_mask.to(device),
        n_steps=args.n_steps,
        method="pndm",
        scale=scale.to(device) if net.use_global else None,
    ).cpu()

    pt_mean = scale[:, 0].view(-1, 1)
    pt_std = scale[:, 1].view(-1, 1)
    eta_mean = scale[:, 2].view(-1, 1)
    eta_std = scale[:, 3].view(-1, 1)
    phi_mean = scale[:, 4].view(-1, 1)
    phi_std = scale[:, 5].view(-1, 1)
    tr_mask = mask[..., 0]
    pf_mask = mask[..., 1]
    if fs_mask is None:
        fs_mask = pf_mask
    l_pt_tr[n : truth.shape[0] + n] = (truth[..., 0] * pt_std + pt_mean).exp().numpy()
    l_eta_tr[n : truth.shape[0] + n] = (truth[..., 1] * eta_std + eta_mean).numpy()
    l_phi_tr[n : truth.shape[0] + n] = reshape_phi(
        truth[:, :, 2] * phi_std + phi_mean
    ).numpy()
    l_class_tr[n : truth.shape[0] + n] = (tr_mask).float().numpy()
    l_ind_tr[n : truth.shape[0] + n] = (tr_mask).float().numpy()

    l_pt_pf[n : truth.shape[0] + n] = (pflow[..., 0] * pt_std + pt_mean).exp().numpy()
    l_eta_pf[n : truth.shape[0] + n] = (pflow[..., 1] * eta_std + eta_mean).numpy()
    l_phi_pf[n : truth.shape[0] + n] = reshape_phi(
        pflow[:, :, 2] * phi_std + phi_mean
    ).numpy()
    l_class_pf[n : truth.shape[0] + n] = (pf_mask).float().numpy()
    l_ind_pf[n : truth.shape[0] + n] = (pf_mask).float().numpy()

    l_pt_fs[n : truth.shape[0] + n] = (fs[..., 0] * pt_std + pt_mean).exp().numpy()
    l_eta_fs[n : truth.shape[0] + n] = (fs[..., 1] * eta_std + eta_mean).numpy()
    l_phi_fs[n : truth.shape[0] + n] = reshape_phi(
        fs[:, :, 2] * phi_std + phi_mean
    ).numpy()
    l_class_fs[n : truth.shape[0] + n] = (fs_mask).float().numpy()

    l_ind_fs[n : truth.shape[0] + n] = (fs_mask).float().numpy()
    n += truth.shape[0]
with uproot.recreate(eval_path) as file:
    file["truth_tree"] = {
        "tr_pt": ak.Array(l_pt_tr),
        "tr_eta": ak.Array(l_eta_tr),
        "tr_phi": ak.Array(l_phi_tr),
        "tr_ind": ak.Array(l_ind_tr),
        "tr_class": ak.Array(l_class_tr),
    }
    file["pflow_tree"] = {
        "pf_pt": ak.Array(l_pt_pf),
        "pf_eta": ak.Array(l_eta_pf),
        "pf_phi": ak.Array(l_phi_pf),
        "pf_ind": ak.Array(l_ind_pf),
        "pf_class": ak.Array(l_class_pf),
    }
    file["fastsim_tree"] = {
        "fs_pt": ak.Array(l_pt_fs),
        "fs_eta": ak.Array(l_eta_fs),
        "fs_phi": ak.Array(l_phi_fs),
        "fs_ind": ak.Array(l_ind_fs),
        "fs_class": ak.Array(l_class_fs),
    }
    if "cocoa" not in config["name"]:
        file["jet_tree"] = {key: val[:n] for key, val in jet_data.items()}
