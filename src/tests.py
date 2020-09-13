import scipy.io
from keras import backend as K

# from RemovedFiles.inception_blocks_v2 import InceptionModel
from RemovedFiles.fr_utils import load_weights, img_to_encoding,load_weights_from_FaceNet
from process import log
from Model.InceptionV2 import InceptionModuleBuilder
from FaceRecognize import FaceRecognizer

K.set_image_data_format('channels_last')


# def test1():
#     print('creating model...', end='')
#     model = InceptionModel((3, 96, 96))
#     print('OK')
#     print('loading weights...', end='')
#     weights = load_weights()
#     layers = [
#         'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3'
#     ]
#     for l in layers:
#         model.get_layer(l).set_weights(weights[l])
#     print('OK')
#     enc = img_to_encoding('faces/andrew.jpg', model)
#     scipy.io.savemat('test1_data.mat', {'enc': enc})
#     log(lambda: print(enc))
#
#
# def modelInitialLayersTest():
#     print('creating model...', end='')
#     builder = InceptionModuleBuilder()
#     model = builder.BuildInception()
#     print('OK')
#     log(lambda: weight_info(model))
#     print('loading weights...', end='')
#     weights = load_weights()
#     layers = [
#         'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3'
#     ]
#     for l in layers:
#         model.get_layer(l).set_weights(weights[l])
#     print('OK')
#     enc = img_to_encoding('faces/andrew.jpg', model)
#     scipy.io.savemat('test1_data.mat', {'enc': enc})
#     log(lambda: print(enc))
#
#
# def weight_info(model):
#     w = model.get_layer('conv1').get_weights()
#     for weight in w:
#         print('shape: {}'.format(weight.shape))
#         print(weight)


def inception_model_test():
    print('creating model...', end='')
    builder = InceptionModuleBuilder()
    model = builder.BuildInception()
    print('OK')
    model.summary()


def inception_model_load_weight_test():
    builder = InceptionModuleBuilder()
    print('creating model...', end="")
    model = builder.BuildInception()
    print('Ok')
    print('loading weights...', end='')
    load_weights_from_FaceNet(model)
    print('Ok')


def Encoding_test():
    rec = FaceRecognizer()
    enc = rec.add_target('faces/andrew.jpg', 'andrew')


def add_target_test():
    rec = FaceRecognizer()
    rec.add_target('faces/arnaud.jpg', 'arnaud')
    mat = scipy.io.loadmat('data_base.mat')
    assert 'arnaud' in mat.keys()


def net_weight_test():
    rec = FaceRecognizer()
    print(rec.calculate_distance('faces/camera_0.jpg', 'faces/younes.jpg'))
    
    # with InceptionV1 need to train again
    # cam0 and ardrew: 0.48233244(dif)
    # cam0 and younes: 0.27634242(same)
    # cam1 and bertrand: 0.21346937
    # cam5 and arnaud: 0.1799758
    # cam2 and arnaud: 0.26394957(dif)

    # with InceptionV1 from coursera course         with InceptionV2 modified
    # cam0 and andrew: 0.9958899                    cam0 and younes: 0.6039957(same)
    # cam2 and arnaud: 0.74894875                   cam0 and andrew: 0.80144095(dif)
    # cam0 and younes: 0.6671407                    can2 and arnaud: 0.5237957(dif nearly same)
    # cam1 and bertrand: 0.46807346                 cam1 and bertrand: 0.6565445(same)


net_weight_test()
# TODO: chack whether the problem is in naming the layers



