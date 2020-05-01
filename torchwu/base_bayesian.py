import torch.nn as nn


class BayesianModule(nn.Module):

    """Base class for BNN to enable certain behaviour."""

    def __init__(self):
        super().__init__()

    def kld(self, *args):
        raise NotImplementedError('BayesianModule::kld()')
