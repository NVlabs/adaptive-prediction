import numpy as np
import torch

from trajectron.model.components.gmm2d import GMM2D


def test_gmm2d():
    # Tensors are of shape (S, B, T, K[, D])
    dist = GMM2D(
        log_pis=torch.full((1, 1, 1, 2), fill_value=0.5),
        mus=torch.concat(
            [torch.zeros((1, 1, 1, 1, 2)), torch.ones((1, 1, 1, 1, 2))], dim=-2
        ),
        log_sigmas=-torch.ones((1, 1, 1, 2, 2)) * 2,
        corrs=torch.zeros((1, 1, 1, 2)),
    )

    print()
