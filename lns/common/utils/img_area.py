"""Utility functions for computing image area without loading entire image."""

import os
import struct


def img_area(img_name: str) -> float:
    """Determine dimensions of image stored at absolute path <img_name>."""
    width, height = _get_image_size(img_name)
    return width * height


# https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
# https://raw.githubusercontent.com/scardine/image_size/master/get_image_size.py
class UnknownImageFormat(Exception):
    """Exception for parsing metadata of image in unknown format."""


def _get_image_size(file_path):  # pylint: disable-msg=too-many-branches
    """Return (width, height) for a given img file content w/ no external dependencies."""
    size = os.path.getsize(file_path)

    with open(file_path, "rb") as input_file:
        height = -1
        width = -1
        data = input_file.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            raw_w, raw_h = struct.unpack("<HH", data[6:10])
            width = int(raw_w)
            height = int(raw_h)
        elif ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            raw_w, raw_h = struct.unpack(">LL", data[16:24])
            width = int(raw_w)
            height = int(raw_h)
        elif (size >= 16) and data.startswith(b'\211PNG\r\n\032\n'):
            # older PNGs?
            raw_w, raw_h = struct.unpack(">LL", data[8:16])
            width = int(raw_w)
            height = int(raw_h)
        elif (size >= 2) and data.startswith(b'\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input_file.seek(0)
            input_file.read(2)
            binary = input_file.read(1)
            try:
                while binary and ord(binary) != 0xDA:
                    while ord(binary) != 0xFF:
                        binary = input_file.read(1)
                    while ord(binary) == 0xFF:
                        binary = input_file.read(1)
                    if (ord(binary) >= 0xC0 and ord(binary) <= 0xC3):
                        input_file.read(3)
                        raw_h, raw_w = struct.unpack(">HH", input_file.read(4))
                        break
                    input_file.read(int(struct.unpack(">H", input_file.read(2))[0]) - 2)
                    binary = input_file.read(1)
                width = int(raw_w)
                height = int(raw_h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as ex:
                raise UnknownImageFormat(ex.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height
