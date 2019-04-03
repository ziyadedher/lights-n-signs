"""Shared data preprocessing.

This package manages all data preprocessing and generation of standardized
serialized data formats.

More advanced and specific preprocessing steps will be taken by independent
algorithm generation techniques, but this package provides a common interface
for those methods to be agnostic of the data format itself.
"""

# Import all preprocessing modules so that they are registered in the preprocessor
from lns.common.preprocessing import bosch  # noqa
from lns.common.preprocessing import lights  # noqa
from lns.common.preprocessing import lisa  # noqa
from lns.common.preprocessing import mturk  # noqa
from lns.common.preprocessing import scale_lights # noqa
from lns.common.preprocessing import scale_signs # noqa
