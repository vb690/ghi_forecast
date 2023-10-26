from tqdm import trange

class TorchModelFitter:
    """Utility class for fit torch models
    """

    def __init__(self, device, train_data_loader, test_data_loader):
        self._device = device
        self._train_data_loader = train_data_loader
        self._test_data_loader = test_data_loader

    
    def fit(self, model, epochs, loss_function, optimizer):


        for epoch in trange(epochs):
            pass


    
    def _fit_step(self, model, optimizer, loss_function, regularisation_function, regularisation_strength):
        
        for batch, data in enumerate(self._train_data_loader):

            X, y = data
            optimizer.zero_grad()
            yhat = model(X)

            objective_loss = loss_function(yhat, y)

            parameters = [param for name, param in model.named_parameters() if 'bias' not in name]
            regularisation_loss = regularisation_function(parameters) * regularisation_strength

            loss = objective_loss + regularisation_loss
            
            loss.backward()

        optimizer.step()
