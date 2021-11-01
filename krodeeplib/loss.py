# -*- coding: utf-8 -*-
"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network.
"""
import numpy as np
from krodeeplib.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
        
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
        
class MSE(Loss):
    """
    MSE is mean squared error, although we're
    just going to do totaled squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return 2*(predicted - actual)
    