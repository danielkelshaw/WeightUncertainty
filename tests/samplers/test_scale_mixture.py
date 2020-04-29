import pytest
import torch

from torchwu.samplers.scale_mixture import ScaleMixture


class TestScaleMixture:

    @pytest.fixture
    def scale_mixture(self):

        pi = 0.5
        sigma1 = 1.0
        sigma2 = 0.005

        return ScaleMixture(pi, sigma1, sigma2)

    def test_log_prior(self, scale_mixture):

        w = torch.tensor([0.5], dtype=float)
        log_prob = scale_mixture.log_prior(w)

        assert isinstance(log_prob, torch.Tensor)
