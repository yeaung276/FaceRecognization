import cv2
import ffmpeg
from FaceDetect import FaceDetector
from FaceRecognize import FaceRecognizer

stream = True


def main():

    video = cv2.VideoCapture('test_images/test_video.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    fd = FaceDetector()
    fr = FaceRecognizer()

    if not video.isOpened():
        print('cannot open file!..')
        exit()
    fr.start(confident_threadshold=0.6)

    while True:
        ret, frame = video.read()
        if not ret:
            print('error reading video!...')
            break
        img, face_slices, faces = fd.detectFaces(frame, draw=False)
        if len(face_slices) > 0:
            encs = fr.get_encodings(face_slices)
            proc_encs = fr.process_encoding(encs)
            n, _ = fr.find_on_batch(proc_encs)

            fd.label_faces(img, n, faces)

        if stream is True:
            cv2.imshow('video', img)
            if cv2.waitKey(1) == ord('q'):
                break

        writer.write(img)

    video.release()
    cv2.destroyAllWindows()


def addAudio():
    input_video = ffmpeg.input("output.avi")
    input_audio = ffmpeg.input("test_images/test_video.mp4").audio
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output('output_with_audio.mp4').run()

# fd = FaceDetector(haarcascade_path='Resources/haarcascades/haarcascade_frontalface_default.xml')
# fr = FaceRecognizer()
# fr.add_target('test_images/test(jnd).jpeg', 'johnny_deep')
# fr.add_target('test_images/arnaud.jpg', 'arnaud')
# fr.start(confident_threadshold=0.8)
# print('.')
# print(fr.names)
#
# print(fr.encs.shape)
# gp_img, face_slices, _ = fd.detectFaces(image_path='test_images/GibbsAWE.png', draw=True)
# cv2.imshow('hello', face_slices[0])
# cv2.waitKey(10000)
# encs = fr.get_encodings(face_slices)
# proc_encs = fr.process_encoding(encs)
# n, p = fr.find_on_batch(proc_encs)
# print(p)
# print(n)

def addFace():
    fr = FaceRecognizer(target_database_path='Resources/faces/database.pkl')
    fr.add_target('test_images/test(jsp).jpg', 'Jack Sparrow')
    fr.add_target('test_images/borbosa.jpg', 'Barbosa')
    fr.add_target('test_images/GibbsAWE.png', 'Mr Gibbs')
    fr.add_target('test_images/willturner.jpg', 'Will Turner')
    fr.add_target('test_images/elizabet.jpeg', 'Elizabet Swam')

# addFace()
# main()
addAudio()




