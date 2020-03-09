import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from trainer import Trainer

class MyModel(nn.Module):

    def __init__(self, n_input, n_output):
        super(MyModel, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.fc1 = nn.Linear(self.n_input, self.n_input // 2)
        self.fc2 = nn.Linear(self.n_input // 2, self.n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def loss(self, y, y_pred):
        criterion = nn.MSELoss(reduction="mean")
        return criterion(y, y_pred)

class SimpleData(Dataset):

    def __init__(self, n_individuals=100, n_predictor=10):
        self.n_individuals = n_individuals
        self.n_predictor = n_predictor
        self.X = torch.rand(self.n_individuals, self.n_predictor)
        self.y = torch.sum(self.X, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index].unsqueeze(-1)

train_data = SimpleData(n_individuals=1000, n_predictor=10)
valid_data = SimpleData(n_individuals=100, n_predictor=10)

model = MyModel(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model.__class__.__name__)
print(model.__class__.__init__)
trainer = Trainer(model, train_data, valid_data, optimizer, None, None)
trainer.fit(n_epochs=100)

