import cv2
import PIL


def img_show(img, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is PIL.Image.Image:
        img.show()
    else:
        cv2.imshow('', img)
        cv2.waitKey(0)
