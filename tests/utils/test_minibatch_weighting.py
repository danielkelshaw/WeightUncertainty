import pytest
from torchwu.utils.minibatch_weighting import minibatch_weight


class TestMinibatchWeighting:

    @pytest.mark.parametrize('idx', [0, 5, 9])
    def test_minibatch_weight(self, idx):

        n_batches = 10

        ret_weight = minibatch_weight(idx, n_batches)
        assert isinstance(ret_weight, float)

        if idx == 0:
            assert ret_weight == 1.0
        elif idx == 5:
            assert ret_weight == pytest.approx(0.0314, 1e-3)
        elif idx == 10:
            assert ret_weight == pytest.approx(0.00197, 1e-3)
