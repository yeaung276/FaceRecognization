### Note
I used Siamese net idea and trained with binary cross entrypy loss. You should try train using tripplet loss. The Shared Base Model is Face net from [this link](https://github.com/iwantooxxoox/Keras-OpenFace). The dataset is [here](http://www.cfpw.io/). The model is not impressive but not bad and have a lot of rooms to improve.

### Net Model

![model](https://github.com/yeaung276/FaceRecognization/blob/master/src/Resources/model/model.png)

### Train Model History

![history](https://github.com/yeaung276/FaceRecognization/blob/master/src/Resources/model/model_train_history.png)

### Dataset

http://www.cfpw.io/

### Base Face Data for Recognization

![jack sparrow](https://github.com/yeaung276/FaceRecognization/blob/master/src/test_images/test(jsp).jpg)<br>
![borbosa](https://github.com/yeaung276/FaceRecognization/blob/master/src/test_images/borbosa.jpg)<br>
![will turner](https://github.com/yeaung276/FaceRecognization/blob/master/src/test_images/willturner.jpg)<br>
![elizabat](https://github.com/yeaung276/FaceRecognization/blob/master/src/test_images/elizabet.jpeg)<br>
![mr gibbs](https://github.com/yeaung276/FaceRecognization/blob/master/src/test_images/GibbsAWE.png)<br>

### Recognized Video
![output_optimized](https://user-images.githubusercontent.com/58524393/97069994-65de3200-15fa-11eb-9aa4-50bd0823228b.gif)<br>
Not good but ok XD [click here to download full video](https://github.com/yeaung276/FaceRecognization/blob/master/src/output_with_audio.mp4)


TODO:
find out how can I view pipeline run matadata for the cofiguration used for each component like split ratio, split path, validation results, etcs