import { CascadeClassifier, Mat, RectVector, VideoCapture } from "mirada/dist/src/types/opencv";

const CASCADE_WEIGHTS = 'haarcascade_frontalface_default.xml'

export default class Detector {
    src: Mat | null = null

    gray: Mat

    faces: RectVector

    classifier: CascadeClassifier | null = null

    capture: VideoCapture | null = null

    constructor() {
        this.gray = new cv.Mat();
        this.faces = new cv.RectVector();
    }

    prepare(source: HTMLVideoElement) {
        source.width = source.videoWidth
        source.height = source.videoHeight
        this.classifier = new cv.CascadeClassifier();
        this.classifier.load(CASCADE_WEIGHTS)
        this.src = new cv.Mat(source.height, source.width, cv.CV_8UC4);
        this.capture = new cv.VideoCapture(source)
    }

    detect(targetWidth: number, targetHeight: number) {
        if (this.src == null || this.classifier == null || this.capture == null) {
            throw Error('Detector need to call prepare first')
        }
        this.capture.read(this.src)
        cv.cvtColor(this.src, this.gray, cv.COLOR_RGBA2GRAY, 0);
        this.classifier.detectMultiScale(this.gray, this.faces, 1.1, 4, 0, new cv.Size(150, 150));
        if (this.faces.get(0)) {
            const face = this.faces.get(0)
            const point1 = new cv.Point(face.x, face.y);
            const point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(this.src, point1, point2, [255, 0, 0, 255]);
            const rect = new cv.Rect(face.x, face.y, face.width, face.height)
            const crop = this.src.roi(rect)
            cv.cvtColor(crop, crop, cv.COLOR_RGBA2RGB)
            cv.resize(crop, crop, new cv.Size(targetHeight, targetWidth), 0, 0, cv.INTER_AREA)
            return [this.src, crop] as const
        }
        return [this.src, null] as const
    }

    destroy() {
        this.src?.delete();
        this.classifier?.delete();
        this.gray.delete();
        this.faces.delete();
    }
}
