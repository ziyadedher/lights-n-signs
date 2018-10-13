"""Generation of synthetic datasets for haar training.

This module contains a procedure for creating synthetic datasets for
haar training from single images through augmentation and combination of
vec files.
"""

from typing import Dict, List

from lns_common.preprocess.preprocessing import Dataset


class SyntheticDataset(Dataset):
    """Specialized container for synthetic haar datasets."""

    __source_path: str
    __num_samples: int
    __samples_multiplier: int

    def __init__(self, name: str,
                 source_path: str,
                 images: Dict[str, List[str]],
                 classes: List[str],
                 num_samples: int,
                 samples_multiplier: int) -> None:
        """Initialize the data structure.

        <name> is a unique name for this dataset.
        <source_path> is the path containing the preliminary samples.
        <images> is a mapping of dataset name to list of absolute paths to the
        images in the dataset.
        <classes> is an indexed list of classes.
        <num_samples> is the number of augmented samples that will be created
        by the Augmentor module.
        <samples_multiplier> is the number of samples per each augmented
        sample that will be created for cascade training - the total number
        of samples will be (num_samples * samples_multiplier).

        Creates a synthetic dataset from preprocessed data. Unlike the regular
        Dataset class, this class does not have any annotations. It also
        requires several additional parameters during initialization,
        which describe the final number of samples created.

        NOTE: In general, this class should be instantiated via the
        preprocessing module, rather than directly.
        """
        super(SyntheticDataset, self).__init__(name, images, classes, {})

        self.__source_path = source_path
        self.__num_samples = num_samples
        self.__samples_multiplier = samples_multiplier

    @property
    def source_path(self) -> str:
        """Return the source path containing the preliminary samples."""
        return self.__source_path

    @property
    def num_samples(self) -> int:
        """Return the number of augmented samples in this dataset."""
        return self.__num_samples

    @property
    def samples_multiplier(self) -> int:
        """Return the number of samples to create from each augmented one."""
        return self.__samples_multiplier
