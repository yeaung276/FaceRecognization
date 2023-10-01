from keras.layers import Input, Subtract, Average, concatenate, Dense, Dropout
from keras.regularizers import l2
from keras.models import Model


############## A network to compute similarity score ###########
def get_similarity_model():
    input1 = Input(128)
    input2 = Input(128)
    average = Average([input1, input2])
    difference = Subtract([input1, input2])
    conc = concatenate([difference, average], axis=1, name='dense_layer_1')
    X = Dense(256, activation='relu', kernel_regularizer=l2(0.02))(conc)
    X = Dropout(0.2)(X)
    X = Dense(256, activation='relu', kernel_regularizer=l2(0.02))(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=[X1, X2], outputs=X, name='upper_model')
        
    