from torch.nn import Sequential, Dropout, Linear, LSTM
from torch import nn


def get_dense_block(input_size, layers, activation_function, dropout_rate):
    """Build a block of linear layers with non linearity and dropout
    """
    block = []
    in_features = input_size
    for layer, out_features in enumerate(layers):

        layer_list = [
            Linear(
                in_features=in_features,
                out_features=out_features,
            ),
            getattr(nn, activation_function)(),
            Dropout(p=dropout_rate)
        ]
        block.extend(layer_list)
    
    return Sequential(*block)
