from FaceDetect import FaceDetector
from FaceRecognize import FaceRecognizer


fd = FaceDetector(haarcascade_path='../Resources/haarcascades/haarcascade_frontalface_default.xml')
fr = FaceRecognizer(target_database_path='../Resources/faces/database.pkl')
fr.add_target('test(jnd).jpeg', 'Jack')
fr.add_target('borbosa.jpg', 'Barbosa')
fr.add_target('mr_gibbs.jpg', 'Gibbs')
fr.add_target('willturner.jpg', 'Will Turner')
fr.add_target('elizabet.jpeg', 'Elizabet')
