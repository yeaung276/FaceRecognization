import cv2
from FaceDetect import FaceDetector
from FaceRecognize import FaceRecognizer

# video = cv2.VideoCapture('test_images/test_video.mp4')
# fd = FaceDetector(haarcascade_path='data/haarcascades/haarcascade_frontalface_default.xml')
# fr = FaceRecognizer()
#
# if not video.isOpened():
#     print('cannot open file!..')
#     exit()
# while True:
#     ret, frame = video.read()
#     if not ret:
#         print('error reading video!...')
#         break
#     gp_img, _, _ = fd.detectFaces(frame, draw=True)
#     cv2.imshow('video', gp_img)
#     if cv2.waitKey(1) == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()


fd = FaceDetector(haarcascade_path='data/haarcascades/haarcascade_frontalface_default.xml')
fr = FaceRecognizer()
gp_img, face_slices, _ = fd.detectFaces(image_path='test_images/group_faces.jpg', draw=True)
encs = fr.get_encodings(face_slices)
proc_encs = fr.process_encoding(encs)
for i in proc_encs:
    print(i.shape)




