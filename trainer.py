import torch
from torch.utils.data import DataLoader

# TODO: callbacks, lr_scheduler
class Trainer:
    """
    Class to easily train pytorch models
    Arguments:
        model: pytorch model
        train_data: pytorch dataset of training data
        valid_data: pytorch dataset of validation data [optional]
        optimizer: pytorch optimizer
        lr_scheduler: pytorch learning rate scheduler
        callbacks: TODO
        metric: metric to follow if different to the loss function 
    """

    def __init__(self, model,
                 train_data,
                 valid_data,
                 optimizer,
                 lr_scheduler=None,
                 callbacks=[],
                 metric=None):
        
        self.model = model # Pytorch model. Model must have a loss method to compute the loss
        self.train_data = train_data # torch dataset
        self.valid_data = valid_data # torch dataset

        self.train_loader = DataLoader(self.train_data, batch_size=8)
        if valid_data:
            self.valid_loader = DataLoader(self.valid_data, batch_size=8)

        self.optimizer = optimizer # torch optimizer (Adam, SGD, exc)
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks
        self.metric = metric 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_losses = [] # epochs training metric
        self.valid_losses = [] # epochs valid metric
    
    def fit(self, n_epochs=10):
        for epoch in range(n_epochs):
            self.train() # Train the model on batches
            if self.valid_data:
                self.eval()
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                        epoch + 1, 
                        self.train_losses[-1],
                        self.valid_losses[-1]
                        ))

    def train(self):
        ###################
        # train the model #
        ###################
        train_loss = 0.0
        self.model.train()
        for i, (X_train_batch, y_train_batch) in enumerate(self.train_loader):
            
            X_train_batch = X_train_batch.to(self.device)
            y_train_batch = y_train_batch.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(X_train_batch)
            loss = self.model.loss(outputs, y_train_batch)
            loss.backward()
            self.optimizer.step()

            # update loss
            train_loss += self.metric(outputs.data, y_train_batch.data).data if self.metric else loss
            
        self.train_losses.append(train_loss / (i + 1) )

    def eval(self):
        ##################
        # valid the model #
        ##################
        valid_loss = 0.0
        self.model.eval()
        for i, (X_valid_batch, y_valid_batch) in enumerate(self.valid_loader):

            X_valid_batch = X_valid_batch.to(self.device)
            y_valid_batch = y_valid_batch.to(self.device)
            outputs = self.model(X_valid_batch)
            loss = self.model.loss(outputs, y_valid_batch)

            valid_loss += self.metric(outputs.data, y_valid_batch.data).data if self.metric else loss
        
        self.valid_losses.append(valid_loss / (i + 1) )

    def predict(self, X_test):
        return self.model(X_test)

    def save_model(self, path):
        file_name = "{}".format(self.model.__name__)
        torch.save(model.state_dict(), filepath)
        pass
    
    @property
    def set_model(self, model):
        self.model = model

    @property
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

