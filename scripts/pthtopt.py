import torch
import torch.nn as nn
import numpy as np
import torch.utils.data

print(torch.__version__)

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

model = torch.load('my_model.pth', map_location=torch.device('cpu'))
model.eval()

# dataset
npixels=128
valdata = np.load('valdata.npy')
vallabels = np.load('vallabels.npy')

valdata /= 100.0
valdata = np.transpose(valdata, (0,3,1,2))

valdata = torch.from_numpy(valdata).float()
vallabels = torch.from_numpy(vallabels).float()

traced_script_module = torch.jit.trace(model, valdata[:1,])

traced_script_module.save("torch_model.pt")
