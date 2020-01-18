import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import time

print(torch.__version__)

# Hyper parameters
num_epochs = 60
num_output = 2
batch_size = 64

# dataset
npixels=128
data = np.load('data.npy')
labels = np.load('labels.npy')
valdata = np.load('valdata.npy')
vallabels = np.load('vallabels.npy')

data /= 100.0
valdata /= 100.0
data = np.transpose(data, (0,3,1,2))
valdata = np.transpose(valdata, (0,3,1,2))

data = torch.from_numpy(data)
labels = torch.from_numpy(labels)
valdata = torch.from_numpy(valdata)
vallabels = torch.from_numpy(vallabels)

train_dataset = torch.utils.data.TensorDataset(data,labels)
val_dataset = torch.utils.data.TensorDataset(valdata,vallabels)

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size,True) # create your dataloader
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size,False) # create your dataloader

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

class ConvNet(nn.Module):
    def __init__(self, num_output=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=8, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=2, stride=2))
        # First fully connected layer 
        self.fc1 = nn.Sequential(nn.Linear(8*8*64, 196), nn.ReLU())
        self.bn1 = nn.Sequential(nn.BatchNorm1d(196))
        self.fc2 = nn.Sequential(nn.Linear(196, 98), nn.ReLU())
        self.bn2 = nn.Sequential(nn.BatchNorm1d(98))
        self.fc3 = nn.Sequential(nn.Linear(98, 11), nn.ReLU())
        self.bn3 = nn.Sequential(nn.BatchNorm1d(11))
        self.fc4 = nn.Sequential(nn.Linear(11, num_output))
#----------------------------------------------------------------------------------------

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 8*8*64)
        out = self.fc1(out)
        out = self.bn1(out)
        out = nn.functional.dropout(out,0.5)
        out = self.fc2(out)
        out = self.bn2(out)
        out = nn.functional.dropout(out,0.5)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.fc4(out)
        return out

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_output).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
print(model)

#----------------------------------------------------------------------------------------

def train():
    model.train()
    loss_all = 0
    loss_func = torch.nn.MSELoss()
    for images,labels in train_loader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        out = model(images)
        loss = loss_func(out, labels)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

#----------------------------------------------------------------------------------------
f1 = open("pred.txt", 'ab')
def test(loader, flag=0):
    model.eval()
    correct = 0
    loss_all = 0
    loss_func = torch.nn.MSELoss()
    for images,labels in loader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        pred = model(images)
        if(flag):
            np.savetxt(f1,pred.cpu().detach().numpy())
        loss = loss_func(pred, labels)
        loss_all += batch_size * loss.item()
    return loss_all / len(loader.dataset)

#----------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    print(time.asctime(time.gmtime()))
    thisloss = train()
    train_acc = test(train_loader, 0)
    val_acc  = test(val_loader, 0)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.
        format(epoch, thisloss, train_acc, val_acc))

data=1
valdata=1
testdata = np.load('testdata.npy')
testlabels = np.load('testlabels.npy')
testdata /= 100.0
testdata = np.transpose(testdata, (0,3,1,2))
testdata = torch.from_numpy(testdata)
testlabels = torch.from_numpy(testlabels)
test_dataset = torch.utils.data.TensorDataset(testdata,testlabels)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size,False) # create your dataloader

test_acc  = test(test_loader, 1)
print('Test Acc: {:.5f}'.
    format(test_acc))

model.eval()
torch.save(model, 'my_model.pth')
