"""Generation of synthetic datasets for haar training.

This module contains a procedure for creating synthetic datasets for
haar training from single images through augmentation and combination of
vec files.
"""

from typing import List, Tuple
from Augmentor import Pipeline
from glob import glob
from preprocessing.mergevec.mergevec import merge_vec_files
import cv2
import os
import subprocess


class SyntheticDataset:
    """Specialized container for synthetic haar datasets."""

    __source_path: str

    def __init__(self, name: str,
                 source_path: str,
                 classes: List[str]) -> None:
        """Initialize the data structure.

        <name> is a unique name for this dataset.
        <source_path> is the path containing the preliminary samples.
        <images> is a mapping of dataset name to list of absolute paths to the
        images in the dataset.
        <classes> is an indexed list of classes.

        Creates a synthetic dataset from preprocessed data. Unlike the regular
        Dataset class, this class does not have any annotations. It also
        requires several additional parameters during initialization,
        which describe the final number of samples created.

        NOTE: In general, this class should be instantiated via the
        preprocessing module, rather than directly.
        """
        self.__source_path = source_path

    @property
    def source_path(self) -> str:
        """Return the source path containing the preliminary samples."""
        return self.__source_path

    def generate(self,
                 num_samples: int,
                 multiplier: int = 5) -> Tuple[str, str, str]:
        """
        Preprocess and generate a synthetic dataset for haar cascade training.

        <num_samples> is the number of augmented samples that will be created.

        Creates a set of augmented samples from those found in the source
        directory specified, and returns the paths to files required for
        training. This is done through probabilistic augmentation via the
        Augmentor module. The further application of opencv_createsamples to
        each of the augmented samples created will occur later, during haar
        model setup. Raises `FileNotFoundError` if folder is empty or images
        aren't found.

        NOTE: In order for this function to run properly the folder
        specified durign the initialization of the synthetic dataset
        (also known as `path_to_samples` here) needs to contain NOTHING but
        3 or 4 samples of the logo/sign, scaled to different dimensions.
        """
        p = Pipeline(self.source_path)
        if len(p.augmentor_images) == 0:
            raise ValueError("Source folder must have a few images" +
                             "and nothing else!")

        # Operations
        p.skew_tilt(0.2)
        p.random_distortion(0.15, grid_width=4, grid_height=4, magnitude=1)
        p.skew_corner(0.05)
        p.gaussian_distortion(
            0.2,
            grid_width=4,
            grid_height=4,
            magnitude=3,
            corner="bell",
            method="in",
        )
        p.random_brightness(0.15, 0.1, 2.0)
        p.random_contrast(0.05, 0.1, 2.0)
        p.random_erasing(0.05, 0.11)
        p.greyscale(1.0)

        # Creating augmented samples
        num_augmented = num_samples // multiplier
        print(f"\n\nCreating {num_augmented} augmented samples...\n")
        p.sample(num_augmented)
        imgs = glob(self.source_path + '/output/*')
        print("\n-----------------------------------------------\n")

        # Applying CLAHE
        print("Applying CLAHE to augmented samples...\n")
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        for i in imgs:
            try:
                img = cv2.imread(i)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)
                cv2.imwrite(i, gray)
            except(cv2.error):
                continue
        print("\n-----------------------------------------------\n")

        vecs_dir = os.path.join(self.source_path, "vecs")
        augmented_samples = os.path.join(self.source_path, "output")
        neg_annotations_file = os.path.join(self.source_path, "../bg.txt")
        vector_file = os.path.join(self.source_path, "../positives.vec")
        cascade_folder = os.path.join(self.source_path, "../cascade")

        # Creating haar samples
        os.mkdir(vecs_dir)
        os.mkdir(cascade_folder)
        print("Creating haar samples for each augmented image...\n")
        out = subprocess.run(  # noqa: F841
            [
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "create_samples_multi.sh"
                ),
                augmented_samples,
                vecs_dir,
                str(multiplier),
                str(neg_annotations_file)
            ],
            stdout=subprocess.PIPE
        )
        print("\n-----------------------------------------------\n")

        # Merging vec files
        print("Merging vec files...\n")
        merge_vec_files(vecs_dir, str(vector_file))
        print("\n-----------------------------------------------\n")
        print("READY FOR TRAINING!\n")

        return (vector_file, neg_annotations_file, cascade_folder)
