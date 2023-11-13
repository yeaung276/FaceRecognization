# SSO-Web

The SSO-Web component is responsible for the web frontend of the Single Sign-On (SSO) service. It utilizes Vite for JavaScript and HTML bundling.

## Usage

1. Run `yarn install` to install dependencies.
2. Run `yarn dev` to start the development server.
3. Run `yarn profile` to create TensorFlow.js configuration for custom bundles.
4. Run `yarn optimizeTF` to create optimized TFJS bundles.
5. Run `yarn build` to build the web.

## Summary

- **JavaScript Bundling:** Utilizes Vite for efficient bundling of JavaScript files and HTML pages.
- **Face Detection and Encoding:** Uses OpenCV.js for face detection. Detected faces are then cropped, encoded using MobileNet, and sent to the SSO-Server component.
- **Model Consumption:** Consumes models served by the Model Server using TensorFlow.js GraphModel.


