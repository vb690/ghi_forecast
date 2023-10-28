from torch import nn
from torch.nn import Linear

from modules.models.neural_networks.blocks import get_dense_block


class LinearModel(nn.Module):
    """Model for multivariate regularized regression
    """

    def __init__(self, in_features, out_features, link_function):
        super().__init__()
        self._weights = Linear(
            in_features=in_features, 
            out_features=out_features
        )
        self._link_function = getattr(nn, link_function)()
    
    def forward(self, X):
        z = self._weights(X)
        yhat = self._link_function(z)
        return yhat
    

class MLPModel(nn.Module):
    """Model for multivariate regularized regression
    """

    def __init__(self, in_features, out_features, layers, activation_function, link_function, dropout_rate):
        super().__init__()
        self._dense_block = get_dense_block(
            layers=layers,
            input_size=in_features,
            activation_function=activation_function,
            dropout_rate=dropout_rate
        )
        self._weights = Linear(
            in_features=layers[-1], 
            out_features=out_features
        )
        self._link_function = getattr(nn, link_function)()
    
    def forward(self, X):
        z = self._weights(
            self._dense_block(X)
        )
        yhat = self._link_function(z)
        return yhat