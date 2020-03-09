from torch.utils.data import Dataset

# Custom Dataset wrapper to the Torch Dataset class.
# It should be modified in accordance with your data
class CustomDataset(Dataset):

    def __init__(self, data):

        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]