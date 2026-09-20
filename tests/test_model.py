import torch

from src.model import SurrogateMLP


def test_flux_is_non_negative_and_peak_normalised():
    torch.manual_seed(0)
    model = SurrogateMLP()
    eigenvalue, flux = model(torch.randn(64, model.trunk[0].in_features) * 10)
    assert (flux >= 0).all()
    assert (eigenvalue > 0).all()
    assert torch.allclose(flux.amax(dim=-1), torch.ones(64), atol=1e-5)
