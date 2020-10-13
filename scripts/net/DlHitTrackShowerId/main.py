# main.py

from data import *
from network import *

# This line is important for GPU running, otherwise some weights end up on the CPU
torch.set_default_tensor_type(torch.cuda.FloatTensor)

view = "W"
the_seed = 42
gpu = torch.device('cuda:0')
batch_size=32
NUM_CLASSES = 3   # Standard case with SHOWER = 1, TRACK = 2, NUll = 0

set_seed(the_seed)
bunch = SegmentationBunch(f"Images{view}", "Hits", "Truth", batch_size=batch_size, valid_pct = 0.5, device=gpu)
train_stats = bunch.count_classes(NUM_CLASSES)
weights = get_class_weights(train_stats)

model, loss_fn, optim = create_model(NUM_CLASSES, weights, gpu)

n_epochs = 20
keys = [0, 1, 2]
train_losses = torch.zeros(n_epochs * len(bunch.train_dl), device=gpu)
val_losses = torch.zeros(n_epochs, device=gpu)
batch_losses = torch.zeros(len(bunch.valid_dl), device=gpu)

train_accs = torch.zeros([3, n_epochs * len(bunch.train_dl)], device=gpu)
val_accs = torch.zeros([3, n_epochs], device=gpu)
batch_accs = torch.zeros([3, len(bunch.valid_dl)], device=gpu)

import time
t0 = time.perf_counter()
i = 0
set_seed(the_seed)
for e in range(0, n_epochs):
    model = model.train()
    n_batches = len(bunch.train_dl)
    for b, batch in enumerate(bunch.train_dl):
        x, y = batch
        pred = model.forward(x)
        loss = loss_fn(pred, y)

        train_losses[i] = loss.item()
        train_accs[0][i] = accuracy(pred, y)
        train_accs[1][i] = accuracy(pred, y, type = 1)
        train_accs[2][i] = accuracy(pred, y, type = 2)

        loss.backward()
        optim.step()
        #scheduler.step()
        optim.zero_grad()
        i += 1
        if b == (n_batches - 1):
            save_model(model, x, f"unet_{view}_{e}")

    # Validate
    model = model.eval()
    with torch.no_grad():
        for b, batch in enumerate(bunch.valid_dl):
            x, y = batch
            pred = model.forward(x)
            loss = loss_fn(pred, y)

            batch_losses[b] = loss.item()
            batch_accs[0][b] = accuracy(pred, y)
            batch_accs[1][b] = accuracy(pred, y, type = 1)
            batch_accs[2][b] = accuracy(pred, y, type = 2)
        val_losses[e] = torch.mean(batch_losses)
        val_accs[0][e] = torch.mean(batch_accs[0][~torch.isnan(batch_accs[0])])
        val_accs[1][e] = torch.mean(batch_accs[1][~torch.isnan(batch_accs[1])])
        val_accs[2][e] = torch.mean(batch_accs[2][~torch.isnan(batch_accs[2])])

t1 = time.perf_counter()
print(f"Networked trained in {t1 - t0:0.3f} s")