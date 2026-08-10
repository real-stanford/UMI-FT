import cv2
import numpy as np

def get_image_transform_with_border(in_res, out_res, bgr_to_rgb: bool=False):
    """ adds a border to make the input image square, and then resizes it to the output resolution """
    iw, ih = in_res
    interp_method = cv2.INTER_AREA

    # Determine the size of the square
    size = max(iw, ih)
    top = (size - ih) // 2
    bottom = size - ih - top
    left = (size - iw) // 2
    right = size - iw - left

    def transform(img: np.ndarray):
        assert img.shape == (ih, iw, 3)
        # Add border to make the image square
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        if bgr_to_rgb:
            img = img[:, :, ::-1]
        return img
    
    return transform
