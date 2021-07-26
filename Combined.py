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
        self.fc1 = nn.Linear(9, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

# how the data flows into our NN
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 
    #no non-linear on last layer

  #create instance of the network

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
    net = Net().float()
    train_file = "learnClean.txt"
    # # Instantiate the class
    train_ds = VolFracDataset(train_file)
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    
    #set up for the training to create optimizer and a loss function.
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = nn.MSELoss() #change this 
    for epoch in range(20):
        loss_per_epoch = 0
        jj=0
        # print("\n==============================\n")
        # print("Epoch = " + str(epoch))
        for (batch_idx, batch) in enumerate(train_ldr): #mini batch starting iteration
            # print("\nBatch = " + str(batch_idx))
            X = batch['f'] #inputs
            Y = batch['hk'] #output
            optimizer.zero_grad()
            net_out = net(X.float()).reshape(-1) #pass input data batch into model (forward()called)
            loss = criterion(net_out,Y.float()) #negative log loss between input/output
            loss.backward() #back propagation 
            optimizer.step() # gradient decent 
            loss_per_epoch +=loss.item()
            jj+=1
        loss_per_epoch = loss_per_epoch/jj
        print("Epoch " + str(epoch))       #printing begins
        print(loss_per_epoch)
        
    # pass in the test file here
    test_loss = 0
    correct = 0
    for (batch_idx, batch) in enumerate(train_ldr):
      X = batch['f'] #inputs
      Y = batch['hk'] #output
      net_out = net(X.float()).reshape(-1) 
      test_loss += criterion(net_out, Y.float())
      pred = net_out.data.max() #needs to be fixed
      correct += pred.eq(Y.float()).sum()
      test_loss /= len(train_ldr.dataset)
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_ldr.dataset),
        100. * correct / len(train_ldr.dataset)))

main()
            
