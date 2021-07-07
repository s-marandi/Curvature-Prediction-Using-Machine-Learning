import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


device = T.device("cpu")
# connecting the layers of the diagram, connected neural network layer is represented by the nn.Linear object
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9 * 1, 100)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(100, 1)

# how the data flows into our NN
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x)


# how the data is understood, what is what in the files
class VolFracDataset(T.utils.data.Dataset):
    def __init__(self, src_file):
        x_tmp = np.loadtxt(src_file, usecols=range(1,10))
        y_tmp = np.loadtxt(src_file, usecols=0)

        self.x_data = T.tensor(x_tmp).to(device)
        self.y_data = T.tensor(y_tmp).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
          idx = idx.tolist()
        inp = self.x_data[idx, :]
        out = self.y_data[idx]
        sample = \
          { 'f' : inp, 'hk' : out }
        return sample


def main():
    # File is passed in to be read
    train_file = "learnClean.txt"
    # Instantiate the class
    train_ds = VolFracDataset(train_file)
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=3, shuffle=True)

    for epoch in range(2):
        print("\n==============================\n")
        print("Epoch = " + str(epoch))
        for (batch_idx, batch) in enumerate(train_ldr):
            print("\nBatch = " + str(batch_idx))
            X = batch['f']
            Y = batch['hk']
            print(batch_idx)
            print(Y)
