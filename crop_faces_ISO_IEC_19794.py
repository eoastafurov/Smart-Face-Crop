import dlib
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from PIL import Image
import numpy as np
import cv2
from imutils import face_utils
from skimage import draw

CROP_FACTOR = 4 / 3
WIDTH = 240
HEIGHT = int(WIDTH * CROP_FACTOR)  # = 320
LIST_OF_UNHANDLED_IMAGES = []


def process():
    """
    Crop all images and masks listed in dirs below
    """
    imgs_path = './data/photos/'
    labels_path = './data/annotations/filtered-hair-masks/'
    res_photo_path = './data/cropped-photos_iso_iec/'
    res_labels_path = './data/annotations/cropped-hair-masks_iso_iec/'

    predictor = dlib.shape_predictor("dlibLandmarksModel/shape_predictor_68_face_landmarks.dat")

    if not os.path.exists(res_photo_path):
        os.mkdir(res_photo_path)
    if not os.path.exists(res_labels_path):
        os.mkdir(res_labels_path)

    detector = dlib.get_frontal_face_detector()

    dirs = os.walk(imgs_path)
    for path_from_top, subdirs, files in dirs:
        for f in files:
            filename = str(f).split('.')[0]
            try:
                hair_mask = np.load(labels_path + filename + '.npy')
            except:
                continue

            image_path = str(path_from_top) + '/' + str(f)
            image = imread(image_path)

            # *************************
            resized_image, dets, p = crop_faces_iso(image, detector, predictor, image_path)
            # *************************

            if dets == -1:
                continue

            # *************************
            np_hair_mask = cropLabels(hair_mask, dets, p)
            # *************************

            np.save(res_labels_path + filename, np_hair_mask)
            imsave(res_photo_path + filename + '.png', resized_image)

    print("\n\nUnhandled images list:")
    for img in LIST_OF_UNHANDLED_IMAGES:
        print(img)


def crop_faces_iso(img, detector, predictor, image_path):
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)

    try:
        landmarks = predictor(gray, faces[0])
        landmarks = face_utils.shape_to_np(landmarks)
    except IndexError:
        print("Warning: no faces found in the photo: " + image_path)
        LIST_OF_UNHANDLED_IMAGES.append(image_path)
        return None, -1, None

    p = ParamsIso(img, landmarks)

    # Draw some circles and lines on image
    # draw_landmarks_and_lines(img, p, landmarks)

    cropped_image = img[p.bottom_y:p.top_y, p.left_x:p.right_x]
    resized_image = resize(cropped_image, (HEIGHT, WIDTH))

    return resized_image, faces, p


def cropLabels(hair_mask, dets, p):
    hair_mask = Image.fromarray((hair_mask * 255).astype('uint8'), mode='L')
    area = (p.left_x, p.bottom_y, p.right_x, p.top_y)
    cropped_hair_mask = hair_mask.crop(area)
    resized_hair_mask = cropped_hair_mask.resize((WIDTH, HEIGHT))

    pixels = np.asarray(resized_hair_mask.getdata())
    pixels = np.reshape(pixels, (HEIGHT, WIDTH))

    np_hair_mask = np.zeros((pixels.shape[0], pixels.shape[1]), dtype='uint8')
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            np_hair_mask[i, j] = pixels[i, j] > 0
    return np_hair_mask


def draw_landmarks_and_lines(img, p, landmarks):
    """
    For drawing eye circles and x-parallel lines
    goes through each eye
    """
    # Draw circles on eyes
    rr, cc = draw.circle_perimeter(p.center_of_left_eye_coord[1],
                                   p.center_of_left_eye_coord[0], (landmarks[40][0] - landmarks[37][0]) // 2)
    img[rr, cc, :] = (255, 0, 0)
    rr, cc = draw.circle_perimeter(p.center_of_right_eye_coord[1],
                                   p.center_of_right_eye_coord[0], (landmarks[46][0] - landmarks[43][0]) // 2)
    img[rr, cc, :] = (255, 0, 0)

    # Draw eyes lines
    rr, cc = draw.line(p.center_of_left_eye_coord[1], 0, p.center_of_left_eye_coord[1], img.shape[1] - 1)
    img[rr, cc, :] = (255, 0, 0)
    rr, cc = draw.line(p.center_of_right_eye_coord[1], 0, p.center_of_right_eye_coord[1], img.shape[1] - 1)
    img[rr, cc, :] = (255, 0, 0)


class ParamsIso:
    """
    Image width                         = W
    Image height                        = W / 0.75
    Y coordinate of eyes                = W * 0.6
    X coordinate of First (right) Eye   = W * 0.375
    X coordinate of Second (left) Eye   = (0.625 * W) - 1
    Width from eye to eye (inclusive)   = 0.25 *W
    """
    def __init__(self, img, landmarks):
        self.center_of_left_eye_coord = landmarks[37] + (landmarks[40] - landmarks[37]) // 2
        self.center_of_right_eye_coord = landmarks[43] + (landmarks[46] - landmarks[43]) // 2
        self.x_length_between_center_of_eyes = self.center_of_right_eye_coord[0] - self.center_of_left_eye_coord[
            0]  # 0.25 * W
        self.real_width = self.x_length_between_center_of_eyes * 4
        self.left_x = max(self.center_of_left_eye_coord[0] - int(0.375 * self.real_width) + 1, 0)
        self.right_x = min(self.center_of_right_eye_coord[0] + int(0.375 * self.real_width), img.shape[1])

        self.y_avg = self.center_of_left_eye_coord[1] + (
                    self.center_of_right_eye_coord[1] - self.center_of_left_eye_coord[1]) // 2

        self.bottom_y = max(self.y_avg - int(0.65 * self.real_width), 0)
        self.top_y = min(self.y_avg + int(0.6 * self.real_width + 1), img.shape[0])


if __name__ == "__main__":
    process()
