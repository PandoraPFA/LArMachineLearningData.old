# script_network.py

from data import *
from network import *

# This line is important for ensuring all tensors exist on the same device
torch.set_default_tensor_type(torch.FloatTensor)

view ="W"
filename = f"Output/{view}/unet_{view}_19.pkl"
the_seed = 42
device = torch.device('cpu')
batch_size=32
NUM_CLASSES = 3 # Standard case with SHOWER = 1, TRACK = 2, NUll = 0

set_seed(the_seed)
bunch = SegmentationBunch(f"Images{view}", "Hits", "Truth", batch_size=batch_size, valid_pct = 0.5, device=device)
train_stats = bunch.count_classes(NUM_CLASSES)
weights = get_class_weights(train_stats)

model, loss_fn, optim = load_model(filename, NUM_CLASSES, weights, device)

for batch in bunch.valid_dl:
    x, y = batch
    break
sm = torch.jit.trace(model, x)
sm.save(f"unet_{view}_v1.0.pt")