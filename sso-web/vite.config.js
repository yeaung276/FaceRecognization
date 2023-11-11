import { resolve } from "path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        scan: resolve(__dirname, "scan/index.html"),
      },
    },
  },
  resolve: {
    alias: {
      "@tensorflow/tfjs$": resolve(__dirname, "./custom_tfjs/custom_tfjs.js"),
      "@tensorflow/tfjs-core$": resolve(
        __dirname,
        "./custom_tfjs/custom_tfjs_core.js"
      ),
      "@tensorflow/tfjs-core/dist/ops/ops_for_converter": resolve(
        __dirname,
        "./custom_tfjs/custom_ops_for_converter.js"
      ),
    },
  },
});
