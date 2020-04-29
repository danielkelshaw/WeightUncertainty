def minibatch_weight(batch_idx: int, num_batches: int) -> float:

    """Calculates Minibatch Weight.

    A formula for calculating the minibatch weight is described in
    section 3.4 of the 'Weight Uncertainty in Neural Networks' paper.
    The weighting decreases as the batch index increases, this is
    because the the first few batches are influenced heavily by
    the complexity cost.

    Parameters
    ----------
    batch_idx : int
        Current batch index.
    num_batches : int
        Total number of batches.

    Returns
    -------
    float
        Current minibatch weight.
    """

    return 2 ** (num_batches - batch_idx) / (2 ** num_batches - batch_idx)
