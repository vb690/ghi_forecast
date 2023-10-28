class EarlyStopping:
    """Utility callback class for performing early stopping: training
    will be halted whenever no significant improvement is observed
    for a prolonged period of time.
    """

    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta

        self.tolerance_counter = 0
        self.call_counter = 0

        self.best_loss = 1e8
        self.best_state_dict = None
        self.early_stop = False

    def __call__(self, current_loss, model):
        """On call evaluate if the current loss meets 
        the criteria for early stopping.
        """
        self.call_counter += 1

        if (self.best_loss - current_loss) < self.min_delta:
            self.tolerance_counter += 1
        else:
            self.tolerance_counter = 0
            self.best_loss = current_loss
            self.best_state_dict = {
                "epoch": self.call_counter,
                "model_state": model.state_dict(),
            }

        self.early_stop = self.tolerance_counter > self.tolerance