import torch

from trajectron.model.components import GMM2D


def dist_mode(dist: GMM2D) -> torch.Tensor:
    # Probs are the same across timesteps.
    probs = dist.pis_cat_dist.probs[..., 0, :]
    argmax_probs = probs.reshape(-1, probs.shape[-1]).argmax(dim=-1)
    ml_means = dist.mus[0, torch.arange(argmax_probs.shape[0]), :, argmax_probs]
    return ml_means
