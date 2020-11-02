# network.py

from model import *

import numpy as np
import torch
import torch.optim as opt


def set_seed(seed):
    """Set the various seeds and flags to ensure deterministic performance

        Args:
            seed: The random seed
    """
    torch.backends.cudnn.deterministic = True   # Note, can impede performance
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_class_weights(stats):
    """Get the weights for each class

        Each class has a weight inversely proportional to the number of instances in the training set

        Args:
            stats: The number of instances of each class

        Returns:
            The weights for each class
    """
    weights = 1. / stats
    return [weight / sum(weights) for weight in weights]


def load_model(filename, num_classes, weights, device):
    """Load a model

        Args:
            filename: The name of the file with the pretrained model parameters
            num_classes: The number of classes available to predict
            weights: The weights to apply to the classes
            device: The device on which to run

        Returns:
            A tuple composed (in order) of the model, loss function, and optimiser
    """
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes))
    model.load_state_dict(torch.load(filename, map_location="cpu"))
    model.eval()
    loss_fn = nn.CrossEntropyLoss(torch.as_tensor(weights, device=device, dtype=torch.float))
    optim = opt.Adam(model.parameters())
    return model, loss_fn, optim


def save_model(model, input, filename):
    """Save the model

        The model is saved as both a pkl file and a TorchScript pt file, which can be loaded via
            model.load_state_dict(torch.load(PATH))
            model.eval()

        Args:
            model: The model to save
            input: An example input to the model
            filename: The output filename, without file extension
    """
    eval_model = model.eval()
    torch.save(eval_model.state_dict(), f"{filename}.pkl")
    torch_script_model = torch.jit.trace(eval_model, input, check_trace=False)
    torch_script_model.save(f"{filename}_traced.pt")


def accuracy(pred, truth, type=None):
    """Get the network accuracy

        Args:
            pred: The network prediction
            truth: The true class
            type: The class whose accuracy should be determined (default: None - overall accuracy)

        Returns:
            The accuracy
    """
    target = truth.squeeze(1)
    mask = target != 0 if type is None else target == type
    return (pred.argmax(dim=1)[mask] == target[mask]).float().mean()


def create_model(num_classes, weights, device):
    """Create the model

        Args:
            num_classes: The number of classes available to predict
            weights: The weights to apply to the classes
            device: The device on which to run

        Returns:
            A tuple composed (in order) of the model, loss function, and optimiser
    """
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes))
    loss_fn = nn.CrossEntropyLoss(torch.as_tensor(weights, device=device, dtype=torch.float))
    optim = opt.Adam(model.parameters())
    return model, loss_fn, optim
