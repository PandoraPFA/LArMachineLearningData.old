import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import time
import sys
import MyModelClass

print(torch.__version__)

# Hyper parameters
num_epochs = 1
num_output = 2
batch_size = 64

# dataset
npixels=128
data = np.load('level0/npyfiles/100nue100nufinHitCoords' + str(npixels) + 'trainbipixHitCoords'+sys.argv[1]+'.npy')
labels = np.load('level0/npyfiles/100nue100nufinHitCoords' + str(npixels) + 'trainbipixMCvtx'+sys.argv[1]+'.npy')
valdata = np.load('level0/npyfiles/100nue100nufinHitCoords' + str(npixels) + 'valbipixHitCoords'+sys.argv[1]+'.npy')
vallabels = np.load('level0/npyfiles/100nue100nufinHitCoords' + str(npixels) + 'valbipixMCvtx'+sys.argv[1]+'.npy')

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MyModelClass.ConvNet(num_output).to(device)
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
f1 = open("recobipixvtx"+sys.argv[1]+".txt", 'ab')
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
testdata = np.load('level0/npyfiles/100nue100nufinHitCoords' + str(npixels) + 'testbipixHitCoords'+sys.argv[1]+'.npy')
testlabels = np.load('level0/npyfiles/100nue100nufinHitCoords' + str(npixels) + 'testbipixMCvtx'+sys.argv[1]+'.npy')
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
torch.save(model, 'my_model'+sys.argv[1]+'.pth')
