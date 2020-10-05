import cv2


def resize_image_keep_aspect_ratio(width, image):
    h, w, _ = image.shape
    height = int((width/w) * h)
    shrink_parameter = w/width

    return cv2.resize(image, (width, height)), shrink_parameter
