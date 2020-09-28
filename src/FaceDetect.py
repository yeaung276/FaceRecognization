import cv2
import matplotlib.pyplot as plt


hyperparameters = {
    'scaleFactor': 1.027,
    'minNeighbors': 4
}

LABEL_COLOR = (0, 255, 0)


class FaceDetector:
    def __init__(self, haarcascade_path='haarcascade_frontalface_default.xml'):
        self.harr_cascade_face = cv2.CascadeClassifier(haarcascade_path)

    @staticmethod
    def _load_and_convert(image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), image

    def getFacesCoordinates(self, image_path):
        gray_image, org_image = self._load_and_convert(image_path)
        faces = self.harr_cascade_face.detectMultiScale(
            gray_image,
            scaleFactor=hyperparameters.get('scaleFactor'),
            minNeighbors=hyperparameters.get('minNeighbors')
        )
        return faces, org_image

    def detectFaces(self, image_path, draw=False):
        faces, image = self.getFacesCoordinates(image_path)
        face_slices = []
        for x, y, w, h in faces:
            face_slices.append(cv2.resize(image[y:y+w, x:x+h, :], (96, 96)))
            if draw:
                cv2.rectangle(image, (x, y), (x+w, y+h), LABEL_COLOR, 2)

        return image, face_slices, faces

    @staticmethod
    def label_faces(image, labels, coordinates):
        img = image.copy()
        i = 0
        for x, y, w, h in coordinates:
            cv2.rectangle(img, (x, y), (x+w, y+h), LABEL_COLOR, 2)
            cv2.putText(img, labels[i], (x, y+h), cv2.FONT_HERSHEY_PLAIN, 2, LABEL_COLOR, 2)
            i = i + 1
        return img


# fd = FaceDetector(haarcascade_path='data/haarcascades/haarcascade_frontalface_default.xml')
# gp_img, _, _ = fd.detectFaces(image_path='faces/group_faces.jpg', draw=True)
# plt.imshow(gp_img)
# plt.show()