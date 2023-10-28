class EarlyStopping:
    def __init__(self, tol=10, min_delta=1e-8):
        self.tol = tol
        self.min_delta = min_delta
        self.previous_loss = 1e10
        self.count = 0
        self.stop = False

    def evaluate(self, current_loss):
        if (self.previous_loss - current_loss) <= self.min_delta:
            self.count += 1
            if self.count >= self.tol:
                self.stop = True

        self.previous_loss = current_loss
        return self.stop
