import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
import MyModelClass

print(torch.__version__)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

model = torch.load('my_model.pth', map_location=torch.device('cpu'))
model.eval()

# dataset
valdata = np.load('valdata.npy')
vallabels = np.load('vallabels.npy')

valdata /= 100.0
valdata = np.transpose(valdata, (0,3,1,2))

valdata = torch.from_numpy(valdata).float()
vallabels = torch.from_numpy(vallabels).float()

traced_script_module = torch.jit.trace(model, valdata[:1,])

traced_script_module.save("torch_model.pt")
