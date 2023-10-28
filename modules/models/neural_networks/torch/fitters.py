import numpy as np

from tqdm import tqdm
import torch

class TorchModelFitter:
    def __init__(
        self,
        device="cuda",
    ):
        """Utiliti class for ffitting torch models"""

        self._device = device

    def fit(
        self,
        model,
        optimizer,
        train_loader,
        validation_loader,
        loss_function,
        stopper,
        epochs,
    ):

        pbar = tqdm(range(epochs))
        train_loss_hist = []
        val_loss_hist = []

        for epoch in pbar:

            model, optimizer, epoch_train_loss = self._train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function
            )
            epoch_val_loss = self._validate(
                model=model,
                loss_function=loss_function,
                validation_generator=validation_loader
            )

            train_loss_hist.append(epoch_train_loss)
            val_loss_hist.append(epoch_val_loss)
            pbar.set_description(
                f"Loss {round(epoch_train_loss, 4)} - "
                f"Validation Loss {round(epoch_val_loss, 4)}"
            )

            stopper(current_loss=epoch_val_loss, model=model)
            if stopper.early_stop:
                print(
                    f"Validation did not improve for {stopper.tolerance} epochs, "
                    f"interrupting training."
                )
                break

        model = self._load_best_state(
            model=model, 
            stopper=stopper
        )

        history = {
            "training_loss": train_loss_hist,
            "validation_loss": val_loss_hist
        }

        return model, optimizer, history

    def _train(
        self,
        model,
        optimizer,
        train_loader,
        loss_function
    ):
        training_loss_tracker = 0.0
        model.train(True)
        for batch_number, (batch_X, batch_y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            yhat = model(batch_X)

            training_loss = loss_function(yhat, batch_y)
            training_loss.backward()
            optimizer.step()

            training_loss_tracker += training_loss.item()
        
        avg_training_loss = training_loss_tracker / (batch_number + 1)

        return model, optimizer, avg_training_loss

    def _validate(
        self,
        model,
        loss_function,
        validation_generator,
    ):
        validation_loss_tracker = 0.0
        model.eval()

        with torch.no_grad():
            for batch_number, (batch_X, batch_y) in enumerate(validation_generator):
                yhat = model(batch_X)
                validation_loss = loss_function(yhat, batch_y)
                validation_loss_tracker += validation_loss.item()

        avg_validation_vloss = validation_loss_tracker / (batch_number + 1)
        return avg_validation_vloss

    @staticmethod
    def _load_best_state(model, stopper):
        best_model_state = stopper.best_state_dict["model_state"]
        best_epoch = stopper.best_state_dict["epoch"]
        model.load_state_dict(best_model_state)
        print(f"Loaded the latest best model state from epoch {best_epoch}")
        return model