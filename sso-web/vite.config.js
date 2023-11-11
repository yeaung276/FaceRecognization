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
});
