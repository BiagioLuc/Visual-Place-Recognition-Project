import os
from glob import glob

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        print(f"Finding images within {dataset_folder} using glob()...")
        images_paths = sorted(glob(f"{dataset_folder}/*/.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any .jpg images")

    return images_paths


class TrainDataset(data.Dataset):
    """Dataset class tailored for feature extraction and allocation,
    without geographic coordinates processing or ground truth computation.
    """
    def _init_(self, dataset_folder, image_size=None):
        super()._init_()

        self.dataset_folder = dataset_folder
        self.images_paths = read_images_paths(self.dataset_folder)

        transformations = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))
        self.transform = transforms.Compose(transformations)

    def _getitem_(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.transform(pil_img)
        return normalized_img, index

    def _len_(self):
        return len(self.images_paths)

    def _repr_(self):
        return f"< Dataset class - #images: {len(self.images_paths)} >"