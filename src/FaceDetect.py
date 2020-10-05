import cv2
import matplotlib.pyplot as plt
import time
from Utils.Images import resize_image_keep_aspect_ratio


hyperparameters = {
    'scaleFactor': 1.07,
    'minNeighbors': 5,
    'frame_width': 500,
}

LABEL_COLOR = (0, 255, 0)


class FaceDetector:
    def __init__(self, haarcascade_path='haarcascade_frontalface_default.xml'):
        self.harr_cascade_face = cv2.CascadeClassifier(haarcascade_path)
        self.shrink_parameter = None

    def _load_and_convert(self, image_path):
        if type(image_path) is str:
            image = cv2.imread(image_path)
        else:
            image = image_path
        image_resize, self.shrink_parameter = resize_image_keep_aspect_ratio(hyperparameters['frame_width'], image)

        return cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY), image

    def getFacesCoordinates(self, image_path):
        gray_image, org_image = self._load_and_convert(image_path)
        faces = self.harr_cascade_face.detectMultiScale(
            gray_image,
            scaleFactor=hyperparameters.get('scaleFactor'),
            minNeighbors=hyperparameters.get('minNeighbors'),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces, org_image

    def detectFaces(self, image_path, draw=False):

        faces, image = self.getFacesCoordinates(image_path)
        img_copy = image.copy()
        face_slices = []
        for x, y, w, h in faces:
            x, y, w, h = self.deShrink(x, y, w, h)
            face_slices.append(cv2.resize(image[y:y+w, x:x+h, :], (96, 96)))
            if draw:
                self.draw_rectangle([(x, y, w, h)], img_copy)

        return img_copy, face_slices, faces

    def deShrink(self, x, y, w, h):
        x = int(self.shrink_parameter * x)
        y = int(self.shrink_parameter * y)
        w = int(self.shrink_parameter * w)
        h = int(self.shrink_parameter * h)
        return x, y, w, h

    @staticmethod
    def draw_rectangle(face_coordinates, image):
        for x, y, w, h in face_coordinates:
            cv2.rectangle(image, (x, y), (x+w, y+h), LABEL_COLOR, 2)
        return image

    @staticmethod
    def label_faces(image, labels, coordinates):
        img = image.copy()
        i = 0
        for x, y, w, h in coordinates:
            cv2.rectangle(img, (x, y), (x+w, y+h), LABEL_COLOR, 2)
            cv2.putText(img, labels[i], (x, y+h), cv2.FONT_HERSHEY_PLAIN, 2, LABEL_COLOR, 2)
            i = i + 1
        return img


# start = time.time()
# fd = FaceDetector(haarcascade_path='data/haarcascades/haarcascade_frontalface_default.xml')
# gp_img, _, _ = fd.detectFaces(image_path='test_images/group_faces.jpg', draw=True)
# end = time.time()
# print('time taken: {}'.format(end-start))
# plt.imshow(gp_img)
# plt.show()