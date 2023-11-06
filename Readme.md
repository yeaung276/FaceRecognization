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

done
create end to end production pipeline using tfx which can be run in cloud like vertax ai pipeline
create custom example gen component which produce triplets that can be use to train model in the later stage using apache_beam programming paradiam
create embedding gen component which convert the produced triplet images into embeddings and saved as TFRecord files

create web to inference which use tfjs and opencvjs
use tfprofilling to produced effeicient bundles of tensorflowjs bundles which decrease the loading time, bundle size and increase efficiency

web
run pip install -r requirement.txt
run bash py_js_model_conversion.sh
go to web
run yarn build

run docker-compose up to spin up the services

mocks/example_gen/4/1, 4/2, 3/1
mocks/example_gen/3/1, 3/2, 4/1

mocks/example_gen/1/1, 1/2, 2/2
mocks/example_gen/2/2, 2/1, 1/2