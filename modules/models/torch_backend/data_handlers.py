from torch.utils.data import Dataset


class TimeseriesAutoregressiveDataset(Dataset):
    def __init__(self, X, y, ar_window, forecast_horizon, gap):
        self._X = X
        self._y = y
        self._ar_window = ar_window
        self._gap = gap
        self._forecast_horizon = forecast_horizon

    def __len__(self):
        seq_len = (
            self._X.__len__() - (self._forecast_horizon + self._ar_window)
        ) / self._gap
        return int(seq_len)

    def __getitem__(self, index):
        index = index * self._gap
        end_insample = index + self._ar_window
        end_outsample = end_insample + self._forecast_horizon
        return (self._X[index:end_insample], self._y[end_insample:end_outsample])
