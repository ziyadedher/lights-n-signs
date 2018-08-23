"""Data processing for Haar cascade training.

Manages all data processing for the generation of data ready to be trained
on with OpenCV Haar training scripts.
"""


class HaarData:
    """Data container for all Haar processed data."""

    def __init__(self) -> None:
        """Initialize the data structure."""
        # TODO: implement
        raise NotImplementedError


class HaarProcessor:
    """Haar processor responsible for data processing to Haar-valid formats."""

    def __init__(self) -> None:
        """Initialize the Haar processor."""
        # TODO: implement
        raise NotImplementedError

    def process(self) -> HaarData:
        """Process all required data, store it, and return a pointer to it."""
        # TODO: implement
        raise NotImplementedError

    def generate_annotations(self) -> None:
        """Generate all annotation files needed for Haar training."""
        # TODO: implement
        raise NotImplementedError
