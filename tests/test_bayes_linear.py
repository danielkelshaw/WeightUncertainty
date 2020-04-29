import pytest
import torch

from torchwu.base_bayesian import BayesianModule
from torchwu.bayes_linear import BayesLinear


class TestBayesLinear:

    @pytest.fixture
    def bayes_linear(self):
        return BayesLinear(5, 3)

    def test_forward(self, bayes_linear):

        input = torch.arange(1, 6, dtype=torch.float32)
        ret_tensor = bayes_linear.forward(input)

        assert len(ret_tensor) == 3
        assert isinstance(ret_tensor, torch.Tensor)
        assert isinstance(bayes_linear, BayesianModule)
