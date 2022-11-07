import torch

from trajectron.utils.matrix_utils import mat


def test_mat():
    x = torch.tensor([1, 2, 3, 4])

    assert torch.allclose(
        mat(x, rows=2),
        torch.tensor(
            [[1, 0, 2, 0, 3, 0, 4, 0], [0, 1, 0, 2, 0, 3, 0, 4]], dtype=torch.float
        ),
    )

    x = torch.tensor([1, 2])

    assert torch.allclose(
        mat(x), torch.tensor([[1, 0, 2, 0], [0, 1, 0, 2]], dtype=torch.float)
    )
