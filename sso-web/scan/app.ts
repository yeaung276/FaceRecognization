import { Mat } from "mirada/dist/src/types/opencv";
import Detector from "./detection";
import Encoder, { WIDTH, HEIGHT } from "./encoding";
import { CASCADE_CLASSIFIER_WEIGHTS_PATH, SSO_SERVER_PATH } from "./env";
import { Tensor, tensor } from "@tensorflow/tfjs";

const FPS = 30;
const INFERENCE_PER_SECONOD = 2;

export default class App {
  detector: Detector;

  encoder: Encoder;

  fpsHandler: NodeJS.Timer | null = null;

  currentFrame = 0;

  isSubmitting = false;

  constructor() {
    this.detector = new Detector();
    this.encoder = new Encoder();
  }

  loadResources() {
    const url = CASCADE_CLASSIFIER_WEIGHTS_PATH;
    let request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "arraybuffer";
    request.onload = function () {
      if (request.readyState === 4) {
        if (request.status === 200) {
          let data = new Uint8Array(request.response);
          cv.FS_createDataFile(
            "/",
            "haarcascade_frontalface_default.xml",
            data,
            true,
            false,
            false
          );
        } else {
          console.log("Failed to load " + url + " status: " + request.status);
        }
      }
    };
    request.send();
  }

  loadVideoStream() {
    const source = document.getElementById("video") as HTMLVideoElement;
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then(function (stream) {
          source.srcObject = stream;
        })
        .catch(function (error) {
          console.error("There was an error accessing the camera:", error);
        });
    } else {
      alert("Your browser does not support the getUserMedia API.");
    }
  }

  async start() {
    this.loadResources();
    this.loadVideoStream();
    await this.encoder.prepare();
    document.getElementById("video")?.addEventListener("loadeddata", () => {
      if (INFERENCE_PER_SECONOD > FPS) {
        throw Error(
          "Inference per second cannot be larger than frame per second"
        );
      }
      this.detector.prepare(
        document.getElementById("video") as HTMLVideoElement
      );
      setTimeout(() => this.nextFrame(), 0);
    });
  }

  nextFrame() {
    this.currentFrame = ++this.currentFrame % FPS;
    let begin = Date.now();
    const [output, face] = this.detector.detect(WIDTH, HEIGHT);
    if (this.currentFrame === Math.floor(FPS / INFERENCE_PER_SECONOD)) {
      this.submit(face);
    }
    cv.imshow("canvasOutput", output);
    const delay = 1000 / FPS - (Date.now() - begin);
    this.fpsHandler = setTimeout(() => this.nextFrame(), delay);
  }

  async submit(face: Mat | null) {
    if (this.isSubmitting || !face) {
      return;
    }
    this.isSubmitting = true;
    const embedding = await new Promise<Tensor>((res, _) => {
      const face_tensor = tensor(face.data, [HEIGHT, WIDTH, 3]);
      const result = this.encoder.encode(face_tensor) as Tensor;
      res(result);
    });
    await this.authenticate(embedding);
    // this.isSubmitting = false
  }

  async authenticate(tensor: Tensor) {
    const params = new URLSearchParams(window.location.search);
    if (params.get("type") === "register") {
      const response = await fetch(`${SSO_SERVER_PATH}/register`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify({
          user_name: params.get("username"),
          name: params.get("name"),
          face_embedding: {
            vector: ((await tensor.array()) as number[])[0],
          },
        }),
      });
      if (!response.ok) {
        document.getElementById("error")!.innerHTML =
          (await response.json()).detail ?? "Something went wrong";
        document.getElementById("error")!.hidden = false;
      }
    }
    if(params.get("type")==="login"){
        const response = await fetch(`${SSO_SERVER_PATH}/authenticate`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
            },
            body: JSON.stringify({
              user_name: params.get("username"),
              redirect_uri: params.get('redirect'),
              face_embedding: {
                vector: ((await tensor.array()) as number[])[0],
              },
            }),
          });
          if (!response.ok) {
            document.getElementById("error")!.innerHTML =
              (await response.json()).detail ?? "Something went wrong";
            document.getElementById("error")!.hidden = false;
          }
    }
  }
}
