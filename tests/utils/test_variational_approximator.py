import pytest
import torch
import torch.nn as nn

from torchwu.bayes_linear import BayesLinear
from torchwu.utils.variational_approximator import variational_approximator


class TestVariationalApproximator:

    @pytest.fixture
    def variational_model(self):

        to_feed = torch.ones(5)

        @variational_approximator
        class VariationalApproximator(nn.Module):
            def __init__(self):
                super().__init__()
                self.bayes_linear = BayesLinear(5, 5)

            def forward(self, x):
                return self.bayes_linear(x)

        model = VariationalApproximator()
        output = model(to_feed)

        return model

    def test_kl_divergence(self, variational_model):

        kl_divergence = variational_model.kl_divergence()

        assert hasattr(variational_model, 'kl_divergence')
        assert isinstance(kl_divergence, torch.Tensor)

    def test_elbo(self, variational_model):

        criterion = nn.MSELoss()
        elbo = variational_model.elbo(
            torch.ones(5), torch.rand(5), criterion, 5
        )

        assert hasattr(variational_model, 'elbo')
        assert isinstance(elbo, torch.Tensor)
