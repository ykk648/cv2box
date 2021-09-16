from face_align.mtcnn_insightface.mtcnn import MTCNN
import cv2
import numpy as np
from ai_utils import img_show
from face_align import align_face_ffhq


def mtcnn_detect(img_p):
    mtcnn = MTCNN()
    bboxes, faces = mtcnn.align_multi(img_p, limit=1, min_face_size=30, crop_size=(224, 224))

    print(bboxes)
    face_ = np.array(faces[0])[:, :, ::-1]
    return face_


if __name__ == '__main__':
    face_img_p = 'test_img/rb.jpeg'

    # # mtcnn from insightface
    # # https://github.com/taotaonice/FaceShifter
    # face = mtcnn_detect(face_img_p)

    # ffhq align method
    face = align_face_ffhq(face_img_p, output_size=112)

    img_show(face)
