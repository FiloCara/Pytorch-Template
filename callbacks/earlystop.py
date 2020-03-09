class EarlyStop:

    def __init__(self, loss, patience=50):
        self.loss = loss
        self.patience = patience
        self.best_score = None

    def __call__(self, current_loss):
        pass
        