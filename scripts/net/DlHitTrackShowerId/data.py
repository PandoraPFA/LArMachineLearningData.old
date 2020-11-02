# data.py

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SegmentationDataset(Dataset):
    """Dataset suitable for segmentation tasks.
    """

    def __init__(self, image_dir, mask_dir, filenames, transform=None, device=torch.device('cuda:0')):
        """Constructor.

            Args:
                image_dir: The directory containing the images
                mask_dir: The directory containing the masks
                filenames: The filanems for the images associate with this dataset
                transform: Optional transform to be applied on a sample (default: None).
                device: The device on which tensors should be created (default: 'cuda:0')
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.filenames = filenames
        self.shift = 0.
        self.norm = 255.
        self.normalise = False
        self.device = device

    def set_image_stats(self, shift, norm):
        """Set the image normalisation parameters.

            Applied as norm_val = (val-shift) / norm

            Args:
                shift: The shift parameter (e.g. mean)
                norm: The normalisation parameter (e.g. standard deviation)
        """
        self.shift = shift
        self.norm = norm

    def set_normalisation(self, norm=True):
        """Sets whether or not the image should be normalised.

            Args:
                norm: True if the image should be normalised, False otherwise (default: True)
        """
        self.normalise = norm

    def __len__(self):
        """Retrieve the number of samples in the dataset.

            Returns:
                The number of samples in the dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """Retrieve a sample from the dataset.

            Args:
                idx: The index of the sample to be retrieved

            Returns:
                The sample requested
        """
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        image = np.asarray(self.open_image(img_name)).astype(np.float32)

        mask_name = os.path.join(self.mask_dir, self.filenames[idx])
        # When using categorical cross entropy, need an un-normalised long
        mask = np.asarray(self.open_image(mask_name)).astype(np.int_)

        image = torch.as_tensor(np.expand_dims(image, axis=0), device=self.device, dtype=torch.float)
        if self.normalise:
            image -= self.shift
            image /= self.norm
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.long)

        return (image, mask)

    def open_image(self, path):
        """Retrieve an image.

            Args:
                path: The path of the image

            Returns:
                The image
        """
        from PIL import Image
        img = Image.open(path)
        if img.mode  != 'L':
            img = img.convert('L')
        return img

class SegmentationBunch():
    """Associates batches of training, validation and testing datasets suitable
        for segmentation tasks.
    """

    def __init__(self, root_dir, image_dir, mask_dir, batch_size, valid_pct=0.1,
                 test_pct=0.0, transform=None, device=torch.device('cuda:0')):
        """Constructor.

            Args:
                root_dir: The top-level directory containing the images
                image_dir: The relative directory containing the images
                mask_dir: The relative directory containing the masks
                batch_size: The batch size
                valid_pct: The fraction of images to be used for validation (default: 0.1)
                test_pct: The fraction of images to be used for testing (default: 0.0)
                transform: Any transforms to be applied to the images (default: None)
                device: The device on which tensors should be created (default: 'cuda:0')
        """
        assert((valid_pct + test_pct) < 1.)
        image_dir = os.path.join(root_dir, image_dir)
        mask_dir = os.path.join(root_dir, mask_dir)
        transform = transform
        image_filenames = next(os.walk(image_dir))[2]
        n_files = len(image_filenames)
        valid_size = int(n_files * valid_pct)
        test_size = int(n_files * test_pct)
        train_size = n_files - (valid_size + test_size)
        train_filenames = image_filenames[:train_size]
        valid_filenames = image_filenames[train_size:train_size + valid_size]
        print(valid_filenames[0:10])

        train_ds = SegmentationDataset(image_dir, mask_dir, train_filenames, transform, device)
        train_ds.set_image_stats(*self.image_stats())
        train_ds.set_normalisation(True)
        self.train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

        valid_ds = SegmentationDataset(image_dir, mask_dir, valid_filenames, transform, device)
        valid_ds.set_image_stats(*self.image_stats())
        valid_ds.set_normalisation(True)
        self.valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

        if test_size > 0:
            test_filenames = random_list[train_size + valid_size:]
            test_ds = SegmentationDataset(image_dir, mask_dir, test_filenames, transform, device)
            test_ds.set_image_stats(*self.image_stats())
            test_ds.set_normalisation(True)
            self.test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
        else:
            self.test_dl = None

    def image_stats(self):
        """Retrieve the normalisation statistics.

            Returns:
                The normalisation statistics as the tuple (shift, norm)
        """
        return 0., 255.

    def count_classes(self, num_classes):
        """Count the number of instances of each class in the training set

            Args:
                num_classes: The number of classes in the training set

            Returns:
                A list of the number of instances of each class
        """
        count = np.zeros(num_classes)
        for batch in self.train_dl:
            _, truth = batch
            unique, counts = torch.unique(truth, return_counts=True)
            unique = [ u.item() for u in unique ]
            counts = [ c.item() for c in counts ]
            this_dict = dict(zip(unique, counts))
            for key in this_dict:
                count[key] += this_dict[key]
        return count