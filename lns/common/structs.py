"""General infrastructure file for all things related to data labels.

Contains data structures for bounding boxes and related things.
"""


class Bounds2D:
    """Two-dimensional bounds object.

    Represents two-dimensional bounding box coordinates with assocated helper functions.
    """

    __left: float
    __top: float
    __width: float
    __height: float

    def __init__(self, left: float, top: float, width: float, height: float) -> None:
        """Initialize a two-dimensional bounds object."""
        self.__left = left
        self.__top = top
        self.__width = width
        self.__height = height

    @property
    def left(self) -> float:
        """Get the left x-coordinate of this box."""
        return self.__left

    @property
    def top(self) -> float:
        """Get the top y-coordinate of this box."""
        return self.__top

    @property
    def width(self) -> float:
        """Get the width of this box."""
        return self.__width

    @property
    def height(self) -> float:
        """Get the height of this box."""
        return self.__height

    @property
    def right(self) -> float:
        """Get the right x-coordinate of this box."""
        return self.left + self.width

    @property
    def bottom(self) -> float:
        """Get the bottom y-coordinate of this box."""
        return self.top + self.height

    @property
    def area(self) -> float:
        """Get the area of this box."""
        return self.width * self.height


class Object2D:  # pylint:disable=too-few-public-methods
    """Two-dimensional object bounds.

    Consists of a two-dimensional bounding box and the classes of the object contained within that bounding box.
    """

    bounds: Bounds2D
    class_index: int
    score: float

    def __init__(self, bounds: Bounds2D, class_index: int, score: float = 1) -> None:
        """Initialize a two-dimensional object."""
        self.bounds = bounds
        self.class_index = class_index
        self.score = score


def iou(box1: Bounds2D, box2: Bounds2D) -> float:
    """Calculate the intersection-over-union of two bounding boxes."""
    x_left = max(box1.left, box2.left)
    y_top = max(box1.top, box2.top)
    x_right = min(box1.right, box2.right)
    y_bottom = min(box1.bottom, box2.bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area / float(box1.area + box2.area - intersection_area)
