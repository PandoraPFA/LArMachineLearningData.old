import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import sys
import MyModelClass

print(torch.__version__)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

model = torch.load('my_model'+sys.argv[1]+'.pth', map_location=torch.device('cpu'))
model.eval()

# dataset
npixels=128
valdata = np.load('level0/npyfiles/100nue100nufinHitCoords'+str(npixels)+'valbipixHitCoords'+sys.argv[1]+'.npy')
vallabels = np.load('level0/npyfiles/100nue100nufinHitCoords'+str(npixels)+'valbipixMCvtx'+sys.argv[1]+ '.npy')

valdata /= 100.0
valdata = np.transpose(valdata, (0,3,1,2))

valdata = torch.from_numpy(valdata).float()
vallabels = torch.from_numpy(vallabels).float()

traced_script_module = torch.jit.trace(model, valdata[:1,])

traced_script_module.save("torch_model"+sys.argv[1]+".pt")
