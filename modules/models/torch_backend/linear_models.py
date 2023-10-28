from torch import nn
from torch.nn import Linear


class LinearModel(nn.Module):
    """Model for multivariate regularized regression"""

    def __init__(self, in_features, out_features, link_function):
        super().__init__()
        self._weights = Linear(in_features=in_features, out_features=out_features)
        self._link_function = getattr(nn, link_function)()

    def forward(self, X):
        z = self._weights(X)
        yhat = self._link_function(z)
        return yhat
