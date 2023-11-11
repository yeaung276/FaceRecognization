import Encoder from "./scan/encoding";
import { profile, zeros } from "@tensorflow/tfjs-core";
import { io } from "@tensorflow/tfjs-node";
import { writeFile } from "fs";

async function profileEncoder() {
  const profileInfo = await profile(async () => {
    const encoder = new Encoder();
    await encoder.prepare(
      io.fileSystem("../models/jsmodels/mobile-net/model.json") as never
    );
    const tensor = zeros([112, 112, 3]);
    encoder.encode(tensor);
  });
  return profileInfo;
}

function writeTFConfig(kernals: string[]) {
  const config = {
    kernels: kernals,
    backends: ["cpu"],
    models: ["../models/jsmodels/mobile-net/model.json"],
    outputPath: "./custom_tfjs",
    forwardModeOnly: true,
  };
  writeFile("tfjs-config.json", JSON.stringify(config), () => null);
}

profileEncoder().then((r) => writeTFConfig(r.kernelNames));
