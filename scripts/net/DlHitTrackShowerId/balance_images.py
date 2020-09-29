# balance_images.py
import cv2
import csv
import numpy as np
import os


SHOWER = 1
TRACK = 2

class EventSummary:
    def __init__(self, index, xx, zz, tt, qq):
        """Constructor.

            Args:
                index: The index of the event
                xx: The set of x-coordinates of hits in this event
                zz: The set of z-coordinates of hits in this event
                qq: The set of hit energies in this event
        """
        self.index = index
        self.xx = xx
        self.zz = zz
        self.tt = tt
        self.qq = qq
        self.n_hits = len(tt)
        unique, counts = np.unique(self.tt, return_counts=True)
        cls_freq = dict(zip(unique, counts))
        if SHOWER not in cls_freq:
            cls_freq[SHOWER] = 0
        if TRACK not in cls_freq:
            cls_freq[TRACK] = 0
        self.delta_class = cls_freq[TRACK] - cls_freq[SHOWER]


    def __lt__(self, other):
        """Compare if this EventSummary has a delta_class value less than that of the specified EventSummary.

            Args:
                other: The EventSummary object to compare against

            Returns:
                True if self.delta_class < other.delta_class, False otherwise.
        """
        return self.delta_class < other.delta_class


    def num_hits(self):
        """Return the total number of hits in the event.

            Returns:
                The total number of hits in the event.
        """
        return self.n_hits


class Binning:
    def __init__(self, x_min, x_max, z_min, z_max, block_size, image_width, image_height):
        """Construct the binning for an image.

            Args:
                x_min: The minimum x-coordinate
                x_max: The maximum x-coordinate
                z_min: The minimum z-coordinate
                z_max: The maximum z-coordinate
                block_size: The size of a block in cm
                image_width: The width of the image in pixels
                image_height: The height of the image in pixels
        """
        eps = np.finfo(np.float32).eps
        x_range = (x_max + eps) - (x_min - eps)
        z_range = (z_max + eps) - (z_min - eps)
        n_x = int(np.ceil(x_range / block_size))
        n_z = int(np.ceil(z_range / block_size))

        self.tiles_x = np.linspace(0 - eps, n_x + eps, n_x + 1)
        self.tiles_z = np.linspace(0 - eps, n_z + eps, n_z + 1)
        self.bins_x = np.linspace(0 - eps, block_size + eps, image_width + 1)
        self.bins_z = np.linspace(0 - eps, block_size + eps, image_height + 1)
        self.n_tiles_x = n_x
        self.n_tiles_z = n_z
        self.tile_width = image_width
        self.tile_height = image_height


def find_transition(events):
    """Determine the index where events go from being dominated by hits from showers to hits from tracks.

        Args:
            events: The set of events

        Returns:
            The index for which there are at least as many track-like hits as shower-like hits.
    """
    return next(i for i, event in enumerate(events) if event.delta_class >= 0)


def build_balanced_dataset(showers, tracks):
    """Construct a dataset where the number of shower-like and track-like hits is approximately balanced.

        Args:
            showers: The shower-dominated events to include
            tracks: The track-dominated events to include

        Returns:
            A tuple where the first element is the balanced dataset and the second element is the remaining imbalanced
            set of events.
    """
    balance = 0
    dataset = []
    if len(showers) > len(tracks):
        dataset.append(showers.pop(0))
        balance += dataset[-1].delta_class
    else:
        dataset.append(tracks.pop(0))
        balance += dataset[-1].delta_class
    while len(showers) > 0 and len(tracks) > 0:
        event = showers.pop(0) if balance > 0 else tracks.pop(0)
        dataset.append(event)
        balance += event.delta_class
    residual = tracks if len(tracks) > 0 else showers

    return dataset, residual


def preprocess_file(input_file):
    """Generate summary descriptions for events.

        Args:
            input_file: a CSV file containing event information

        Returns:
            A list of EventSummary objects describing the events.
    """
    events = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data = row[1:-1]
            events.append(preprocess_event(i, data))

    return events


def preprocess_event(index, data):
    """Construct summary event description for a single event.

        The input data has the format:
        N Hits,N*{x coord, z coord, class, charge}

        Args:
            index: the index of the event
            data: the set of vertices and hits for the event

        Returns:
            An EventSummary describing the event.
    """
    n_vals = 4
    n_hits = int(data.pop(0))
    expected_vals = n_hits * n_vals
    observed_vals = len(data)

    if expected_vals > observed_vals:
        print("Missing information in input file")
        print(f"Expected {expected_vals} values, observed {observed_vals} values")
        return
    elif expected_vals < observed_vals:
        print("Excess information in input file")
        print(f"Expected {expected_vals} values, observed {observed_vals} values")
        return

    vals_start, vals_finish = 0, observed_vals
    xx = np.array(data[vals_start:vals_finish:n_vals], dtype=np.float)
    zz = np.array(data[vals_start + 1:vals_finish:n_vals], dtype=np.float)
    tt = np.array(data[vals_start + 2:vals_finish:n_vals], dtype=np.int)
    qq = np.array(data[vals_start + 3:vals_finish:n_vals], dtype=np.float)

    return EventSummary(index, xx, zz, tt, qq)


def make_image(event, output_folder, image_size = (256, 256)):
    """Generate the training/validation set images for a single event.

        Images are output to <output_folder>/Hits and <output_folder>/Truth

        Args:
            event: the EventSummary object
            output_folder: the top-level folder for output images
            image_size: the output image size as a tuple (width, height)
    """
    image_width, image_height = image_size
    x_min, x_max = np.amin(event.xx), np.amax(event.xx)
    z_min, z_max = np.amin(event.zz), np.amax(event.zz)
    q_min, q_max = np.amin(event.qq), np.amax(event.qq)
    x_range = x_max - x_min
    z_range = z_max - z_min
    q_range = q_max - q_min

    if 2 * x_range < image_width:
        padding = 0.5 * (image_width / 2. - x_range)
        x_min -= padding
        x_max += padding
        x_range = x_max - x_min
    if 2 * z_range < image_height:
        padding = 0.5 * (image_height / 2. - z_range)
        z_min -= padding
        z_max += padding
        z_range = z_max - z_min

    block_size = 128
    binning = Binning(x_min, x_max, z_min, z_max, block_size, image_width, image_height)
    eps = np.finfo(np.float32).eps
    x_bins = np.linspace(x_min - eps, x_max + eps, image_width + 1)
    z_bins = np.linspace(z_min - eps, z_max + eps, image_height + 1)

    ptx = np.digitize((event.xx - x_min) / block_size, binning.tiles_x) - 1
    ptz = np.digitize((event.zz - z_min) / block_size, binning.tiles_z) - 1
    px = np.digitize((event.xx - x_min) % block_size, binning.bins_x) - 1
    pz = np.digitize((event.zz - z_min) % block_size, binning.bins_z) - 1

    shower_histogram = np.zeros((binning.n_tiles_z, binning.n_tiles_x, binning.tile_height, binning.tile_width), 'uint32')
    track_histogram = np.zeros((binning.n_tiles_z, binning.n_tiles_x, binning.tile_height, binning.tile_width), 'uint32')
    class_histogram = np.zeros((binning.n_tiles_z, binning.n_tiles_x, binning.tile_height, binning.tile_width), 'uint32')
    temp_histogram = np.zeros((binning.n_tiles_z, binning.n_tiles_x, binning.tile_height, binning.tile_width), 'uint32')

    for idx in range(event.n_hits):
        if event.tt[idx] == SHOWER:
            shower_histogram[ptz[idx], ptx[idx], (image_height - 1) - pz[idx], px[idx]] += event.qq[idx]
        else:
            track_histogram[ptz[idx], ptx[idx], (image_height - 1) - pz[idx], px[idx]] += event.qq[idx]

    for idx in range(event.n_hits):
        temp_histogram[ptz[idx], ptx[idx], (image_height - 1) - pz[idx], px[idx]] += event.qq[idx]

    truth_histogram = np.zeros((binning.n_tiles_z, binning.n_tiles_x, binning.tile_height, binning.tile_width), 'uint8')
    shower_mask = shower_histogram > track_histogram
    track_mask = track_histogram > shower_histogram
    truth_histogram = np.zeros((binning.n_tiles_z, binning.n_tiles_x, binning.tile_height, binning.tile_width), 'uint8')
    truth_histogram[shower_mask] = SHOWER
    truth_histogram[track_mask] = TRACK

    # Normalise input histograms
    q_min, q_max = np.min(temp_histogram), np.max(temp_histogram)
    q_range = q_max - q_min
    temp_histogram = 255. * (temp_histogram - q_min) / q_range

    input_histogram = temp_histogram.astype(np.uint8)
    for tr in range(binning.n_tiles_z):
        for tc in range(binning.n_tiles_x):
            if np.count_nonzero(truth_histogram[tr, tc, ...]) > 10:
                truth_output_folder = os.path.join(output_folder, "Truth")
                truth_filename = os.path.join(truth_output_folder, f"Image_{event.index}_{tr}_{tc}.png")
                cv2.imwrite(truth_filename, truth_histogram[tr, tc, ...])
                hits_output_folder = os.path.join(output_folder, "Hits")
                hits_filename = os.path.join(hits_output_folder, f"Image_{event.index}_{tr}_{tc}.png")
                cv2.imwrite(hits_filename, input_histogram[tr, tc, ...])