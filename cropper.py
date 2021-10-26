import os
import cv2
import numpy as np
import mediapipe as mp
import math
try:
    from estimate_pose import estimate_pose
except ModuleNotFoundError:
    from mp_crop import estimate_pose

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic


class Cropper:
    def __init__(
            self,
            detection_confidence=0.1,
            HEIGHT=320,
            WIDTH=240,
            max_yaw=0.025,
            max_pitch=0.017,
            max_roll=0.34,
            need_to_filtrate=True,
            fill_color='white'
    ):
        self.detection_confidence = detection_confidence
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.need_to_filtrate = need_to_filtrate
        self.unfiltered_images = []
        self.images_without_landmarks = []
        self.fill_color = fill_color

    def get_center_of_right_eye(self, landmarks):
        x = landmarks[380].x + (landmarks[386].x - landmarks[380].x) / 2
        y = landmarks[380].y + (landmarks[386].y - landmarks[380].y) / 2
        return x, y

    def get_center_of_left_eye(self, landmarks):
        x = landmarks[159].x + (landmarks[153].x - landmarks[159].x) / 2
        y = landmarks[153].y + (landmarks[159].y - landmarks[153].y) / 2
        return x, y

    def get_angle_eyes(self, params) -> float:
        """
        if left eye under right eye, then returned positive. Else - negative
        """
        return math.atan((params.right_eye_y - params.left_eye_y) / (params.right_eye_x - params.left_eye_x))

    def filtrate_angles(self, landmarks) -> bool:
        """
        https://cutt.ly/LcNz23T or google
        Accepted values are:
            |yaw|   < 0.025 (10 * sin(15 deg))  ||
            |pitch| < 0.017 (10 * sin(10 deg))  || defined in init as default
            |roll|  < 0.34  (1 * sin(20 deg))   ||
        Returns:
            True    - if angles match criteria above
            False   - if not
        """
        _, yaw, pitch, roll = estimate_pose(landmarks)
        return abs(yaw) < self.max_yaw and abs(pitch) < self.max_pitch and abs(roll) < self.max_roll

    def get_face_rectangle(self, image):
        """
            Get face rectangle by using MP Face Detection.
            originally needed to increase the size of the face as
            a percentage in relation to the total size of the
            photo in order to apply for input to MP FaceMesh,
            who needs close-up faces for work.
        """
        with mp_face_detection.FaceDetection(
                min_detection_confidence=0.01
        ) as face_detection:
            results = face_detection.process(image)
            a = results.detections[0].location_data.relative_keypoints[:2]
            b = a[0].x
        face_coords = [[line[1] for line in
                        results.detections[0].ListFields()[2][1].ListFields()[1][1].ListFields()]]
        imh, imw, _ = image.shape
        x, y, w, h = face_coords[0]
        x, y, w, h = int(imw * x), int(imh * y), int(imw * w), int(imh * h)
        return x, y, w, h

    def get_landmarks_old(self, image):
        """
            Calculates landmarks points on pre-cropped
            via get_face_rectangle() function.

            return:    landmarks in its absolute value
                        (from  0 to image.shape)
        """
        x, y, w, h = self.get_face_rectangle(image.copy())
        face_img = image[y:y + h, x:x + h].copy()
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.1
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks is None:
            return None

        landmarks = results.multi_face_landmarks[0].ListFields()[0][1]
        for landmark in landmarks:
            landmark.x = (landmark.x * w) + x
            landmark.y *= (landmark.y * h) + y
        return landmarks

    def get_landmarks_face_mesh_model(self, image):
        """
            Calculates landmarks on image with a close-up face

            Note:   working only with close-up face
            return: landmarks in its relative values (from 0 to 1)
        """
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.1
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks is None:
                return None
            return results.multi_face_landmarks[0].ListFields()[0][1]

    def get_eyes_coords(self, image):
        """
            Calculates eyes coords provided by MP FaceDetection.

            Note:   working only with close-up face
                    (but not as close-up as FaceMesh requires)
            return: landmarks in its relative values (from 0 to 1)
        """
        with mp_face_detection.FaceDetection(
                min_detection_confidence=0.1
        ) as face_detection:
            results = face_detection.process(image)
            relative_keypoints = results.detections[0].location_data.relative_keypoints[:2]

        left_eye_x, left_eye_y = relative_keypoints[0].x, relative_keypoints[0].y
        right_eye_x, right_eye_y = relative_keypoints[1].x, relative_keypoints[1].y
        return left_eye_x, left_eye_y, right_eye_x, right_eye_y

    def get_landmarks_holistic_model(self, image):
        """
            Calculates landmarks using MP Holistic model.
            Suitable for images where the face is not close-up
            (but is often mistaken in recognition approx from 5 to 20 %).

            return: landmarks in its relative values (from 0 to 1)
        """
        with mp_holistic.Holistic(
                static_image_mode=True,
                upper_body_only=True,
                min_detection_confidence=0.0
        ) as holistic:
            result = holistic.process(image)
            if result.face_landmarks is None:
                return None
            landmarks = np.array(result.face_landmarks.landmark)
        return landmarks

    def get_mp_landmarks(self, image):
        landmarks = self.get_landmarks_face_mesh_model(image.copy())
        if landmarks is None:
            landmarks = self.get_landmarks_holistic_model(image.copy())
            if landmarks is None:
                self.images_without_landmarks.append(image)
                return None
        return landmarks

    def process_image(self, image):
        """
            Empirically it was found that a combination of two functions
            (first is get_landmarks_face_mesh_model(), then if result is None is
            get_landmarks_holistic_model()) give the best result.

            return: Tuple of
                        1. cropped & resized image
                        2. params class
                    or (None, None) if:
                     landmarks is empty or head pose unsuitable
        """
        landmarks = self.get_landmarks_face_mesh_model(image.copy())
        if landmarks is None:
            landmarks = self.get_landmarks_holistic_model(image.copy())
            if landmarks is None:
                self.images_without_landmarks.append(image)
                return None, None

        if not self.filtrate_angles(landmarks):
            self.unfiltered_images.append(image)
            return None, None

        height, width, _ = image.shape

        params = self.Params()
        params.left_eye_x, params.left_eye_y = self.get_center_of_left_eye(landmarks)
        params.right_eye_x, params.right_eye_y = self.get_center_of_right_eye(landmarks)

        params.left_eye_x *= width
        params.right_eye_x *= width
        params.left_eye_y *= height
        params.right_eye_y *= height

        params.length_between_eyes = params.right_eye_x - params.left_eye_x
        params.output_width = params.length_between_eyes * 4
        params.y_avg = params.left_eye_y + (params.right_eye_y - params.left_eye_y) / 2

        params.left_x = int(max(params.left_eye_x - params.output_width * 0.375 + 1, 0))
        params.right_x = int(min(params.right_eye_x + params.output_width * 0.375, width))
        params.bottom_y = int(max(params.y_avg - params.output_width * 0.65, 0))
        params.top_y = int(min(params.y_avg + params.output_width * 0.6 + 1, height))

        cropped_image = image[params.bottom_y:params.top_y, params.left_x:params.right_x]
        resized_image = self.resize_cropped_image_with_borders(cropped_image)

        return resized_image, params

    def convert_mask_to_image(self, mask, key_constant=117):
        blank_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    blank_image[i, j] = [key_constant, key_constant, key_constant]
        return blank_image

    def convert_image_to_mask(self, image_mask, key_constant=117):
        converted = np.zeros((image_mask.shape[0], image_mask.shape[1]), dtype=np.uint8)
        for i in range(image_mask.shape[0]):
            for j in range(image_mask.shape[1]):
                if image_mask[i, j][0] == key_constant:
                    if image_mask[i, j][1] == key_constant:
                        if image_mask[i, j][2] == key_constant:
                            converted[i, j] = 1
        return converted

    def process_image_and_mask(self, image, mask):
        """
            The same as process_image() function,
            but this also crops bit mask.
            Parameters:
                image - np.ndarray of shape (height, width, channels=3)
                mask - np.ndarray of shape  (height, width)
            Returns:
                Tuple of:
                    1. cropped image
                    2. cropped_mask
                or (None, None) if
                    landmarks is empty or head pose unsuitable
        """
        cropped_resized_image, params = self.process_image(image)
        if cropped_resized_image is None:
            return None, None

        mask = mask.reshape(mask.shape[0], mask.shape[1])

        image_mask = self.convert_mask_to_image(mask)
        image_mask_cropped = image_mask[params.bottom_y:params.top_y, params.left_x:params.right_x]
        image_mask_resized = self.resize_cropped_image_with_borders(image_mask_cropped)

        cropped_resized_mask = self.convert_image_to_mask(image_mask_resized)

        return cropped_resized_image, cropped_resized_mask

    def resize_cropped_image_with_borders(self, cropped_image):
        border_v = 0
        border_h = 0

        assert self.fill_color in ['white', 'black']

        _fill_color = [255, 255, 255] if self.fill_color == 'white' else [0, 0, 0]

        assert cropped_image.shape[0] != 0
        assert cropped_image.shape[1] != 0

        if self.HEIGHT / self.WIDTH >= cropped_image.shape[0] / cropped_image.shape[1]:
            border_v = int((((self.HEIGHT / self.WIDTH) * cropped_image.shape[1]) - cropped_image.shape[0]) / 2)
        else:
            border_h = int((((self.WIDTH / self.HEIGHT) * cropped_image.shape[0]) - cropped_image.shape[1]) / 2)

        img_ = cv2.copyMakeBorder(cropped_image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT,
                                  value=_fill_color)
        return cv2.resize(img_, (self.WIDTH, self.HEIGHT))

    class Params:
        """
            Image width                         = W
            Image height                        = W / 0.75
            Y coordinate of eyes                = W * 0.6
            X coordinate of First (right) Eye   = W * 0.375
            X coordinate of Second (left) Eye   = (0.625 * W) - 1
            Width from eye to eye (inclusive)   = 0.25 *W
        """

        def __init__(self):
            # Landmarks params
            self.left_eye_x = None
            self.left_eye_y = None
            self.right_eye_x = None
            self.right_eye_y = None

            # temp params
            self.length_between_eyes = None
            self.output_width = None
            self.y_avg = None

            # output params
            self.left_x = None
            self.right_x = None
            self.bottom_y = None
            self.top_y = None
