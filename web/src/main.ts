import App from './app';

window.onload = () => {
    const app = new App()
    app.start()
}

// function inference(image: any){
//     const source = new cv.Mat()
//     cv.cvtColor(image, source, cv.COLOR_RGBA2RGB)
//     const dist = new cv.Mat();
//     cv.resize(source, dist, new cv.Size(HEIGHT, WIDTH), 0, 0, cv.INTER_AREA)
//     // const tensor = tf.tensor(dist.data, [HEIGHT, WIDTH, 3])
//     // console.log(tensor)
// }
