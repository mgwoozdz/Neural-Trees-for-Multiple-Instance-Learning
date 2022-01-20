"""Collection of functions for images processing"""
import math
import numpy as np


def add_padding(img, padding):
    height, width, channels = img.shape

    new_height = height + padding
    new_width = width + padding

    result = np.full((new_height, new_width, channels), (0, 0, 0), dtype=np.uint8)

    x_center = math.ceil((new_width - width) / 2)
    y_center = math.ceil((new_height - height) / 2)

    result[y_center:y_center + height, x_center:x_center + width] = img

    return result


def crop_patch(x, y, img, padding=None, offset_l=None, offset_r=None):
    x, y = round(x), round(y)
    patch = None

    if padding:
        padded_img = add_padding(img, padding)

        offset = math.ceil(padding / 2)
        x += offset
        y += offset

        x0, x1 = x - offset, x + offset - 1
        y0, y1 = y - offset, y + offset - 1

        try:
            patch = padded_img[x0:x1, y0:y1]
        except IndexError:
            pass
        finally:
            return patch
    else:
        if x < offset_l or x > img.shape[0] - offset_r or y < offset_l or y > img.shape[1] - offset_r:
            return None
        else:
            x0, x1 = x - 13, x + 14
            y0, y1 = y - 13, y + 14

            patch = img[x0:x1, y0:y1]
            return patch