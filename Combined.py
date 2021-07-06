import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
​
​
def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)
​
​
def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
              log_interval=10):
  
​device = T.device("cpu")

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
    train_file = "trainClean.txt"
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
  
​
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(9 * 1, 100)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(100, 1)
​
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)
​
    net = Net()
    print(net)
​
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()
​
    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
​
    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data[0]
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()
​
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
​
​
if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        create_nn()
