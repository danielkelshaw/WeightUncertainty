# **WeightUncertainty**
[![Build Status](https://travis-ci.org/danielkelshaw/WeightUncertainty.svg?branch=master)](https://travis-ci.org/danielkelshaw/WeightUncertainty)

PyTorch implemenation of [**Weight Uncertainty in Neural Networks**](https://arxiv.org/pdf/1907.00865.pdf).

This repository provides an implementation of the `Bayes-by-Backprop` 
framework as described in the [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1907.00865.pdf)
paper. The code provides a simple PyTorch interface and has been
designed to be extensible, allowing for implementation of additional 
functionality such as alternate priors or variational posteriors. In 
addition, modules that employ the *Local Reparameterisation Trick* have 
been implemented to provide the opporunity for faster, stable training.

- [x] Python 3.6+
- [x] MIT License

## **Overview:**
The [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1907.00865.pdf)
*(WUINN)* paper provides a framework known as `Bayes-by-Backprop` which 
allows for learning a probability distribution on the weights of a 
Neural Network. The network weights are regularised by minimising the 
`ELBO` cost given a prior / posterior distribution, this helps avoid the
common pitfalls of conventional Neural Networks where overfitting is a 
major concern and predictions are skewed by an inability to correctly
assess uncertainty in the training data.

The authors of *WUINN* utilise a *Scale Mixture* prior and a
*Gaussian* posterior. This combination is computationally intractable 
and, as such, a variational approximation to the posterior must be 
found - this is completed through Monte Carlo sampling. Due to the
sampling method employed, the parameters of the posterior distribution 
are still capable of being trained through standard gradient descent.
Section 3.2 in the paper provides further information on this.

The minibatching / KL-reweighting scheme mentioned in the paper has been 
implemented to ensure that the first few minibatches are heavily 
influenced by the complexity cost and later batches are predominantly 
influenced by the data. The weights used are simply a function of the 
current batch index and the total number of batches.

This work has been extended through the use of a *Gaussian* distribution 
for the prior and posterior - this allows the *local reparameterisation
trick* (LRT) to be used. As the LRT samples activations rather than the
weights, the KL Divergence can not be calculated through the use of MC
sampling; instead the closed form must be used. Fortunately this is
possible for the *Gaussian* prior / posterior combination.

## **Example:**
An example of a `Bayes-by-Backprop` network has been implemented in
`mnist_bnn.py` - this example can be run with:

```bash
python3 mnist_bnn.py
```

## **References**:

```
@misc{blundell2015weight,
    title={Weight Uncertainty in Neural Networks},
    author={Charles Blundell and Julien Cornebise and Koray Kavukcuoglu and Daan Wierstra},
    year={2015},
    eprint={1505.05424},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

###### PyTorch implementation of [Weight Uncertainty in Neural Networks](https://arxiv.org/pdf/1907.00865.pdf)<br>Made by Daniel Kelshaw
