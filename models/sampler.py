import torch
from tqdm import tqdm


@torch.no_grad()
def transfer(x_curr, d_curr, t_curr, t_next):
    x_next = x_curr + d_curr * (t_next - t_curr)
    return x_next


@torch.no_grad()
def runge_kutta(model, truth, fastsim, mask, scale, t_list, ets, deriv_prev=None):
    e_1 = model.forward(
        fastsim,
        truth,
        mask,
        timestep=t_list[0].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=deriv_prev,
    )
    ets.append(e_1)
    x_2 = transfer(fastsim, e_1, t_list[0], t_list[1])

    e_2 = model.forward(
        x_2,
        truth,
        mask,
        timestep=t_list[1].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=e_1,
    )
    x_3 = transfer(fastsim, e_2, t_list[0], t_list[1])

    e_3 = model.forward(
        x_3,
        truth,
        mask,
        timestep=t_list[1].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=e_2,
    )
    x_4 = transfer(fastsim, e_3, t_list[0], t_list[2])

    e_4 = model.forward(
        x_4,
        truth,
        mask,
        timestep=t_list[2].expand(fastsim.shape[0]),
        scale=scale,
        fs_data_prev=e_3,
    )
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


@torch.no_grad()
def gen_order_4(model, truth, fastsim, mask, scale, t, t_next, ets, deriv_prev=None):
    t_list = [t, (t + t_next) / 2, t_next]
    if len(ets) > 2:
        deriv_ = model.forward(
            fastsim,
            truth,
            mask,
            timestep=t.expand(fastsim.shape[0]),
            scale=scale,
            fs_data_prev=deriv_prev,
        )
        ets.append(deriv_)
        deriv = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        deriv = runge_kutta(
            model,
            truth,
            fastsim,
            mask,
            scale,
            t_list,
            ets,
            deriv_prev=deriv_prev,
        )
    deriv_prev = deriv
    fastsim_next = transfer(fastsim, deriv, t, t_next)
    return fastsim_next, deriv_prev


@torch.no_grad()
def pndm_sampler(model, truth, pflow, mask, scale, n_steps, save_seq=False):
    seq = []
    device = truth.device
    fastsim = torch.randn_like(pflow, device=device)
    fastsim[~mask[..., 1]] = 0
    t_steps = torch.linspace(0, 1, n_steps, device=device)
    t_steps = torch.cat([t_steps, torch.ones_like(t_steps)[:1]]).to(device)
    ets = []
    deriv_prev = torch.zeros_like(pflow, device=device)

    for i, (t_cur, t_next) in tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=n_steps
    ):
        fastsim, deriv_prev = gen_order_4(
            model,
            truth,
            fastsim,
            mask,
            scale,
            t_cur,
            t_next,
            ets,
            deriv_prev=deriv_prev,
        )
        if save_seq:
            seq.append(fastsim.cpu())
    if save_seq:
        seq = torch.stack(seq)
    return fastsim.cpu(), seq
