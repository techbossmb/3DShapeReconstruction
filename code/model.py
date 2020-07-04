import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D, UpSampling3D
from keras.optimizers import SGD
from keras import metrics


def Conv3DInputBlock(model, numfilters, input_shape):
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_1', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool1'))
    return model
    
def Conv3DBlock(model, numfilters, blockid):
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}'.format(blockid)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool_{}'.format(blockid)))
    return model

def DoubleConv3DBlock(model, numfilters, blockid):
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}a'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}b'.format(blockid)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool_{}'.format(blockid)))
    return model

def DoubleConv3DPadBlock(model, numfilters, blockid):
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}a'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}b'.format(blockid)))
    model.add(ZeroPadding3D(padding=((0,0),(0,1),(0,1)), name='zeropad_{}'.format(blockid)))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool_{}'.format(blockid)))
    return model
    
def UpConv3DBlock(model, numfilters, blockid):
    model.add(UpSampling3D(size=(2,2,2), name='pool_{}'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}'.format(blockid)))
    return model

def LastConv3DBlock(model, numfilters, blockid):
    model.add(UpSampling3D(size=(1,1,1), name='pool_{}'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='sigmoid', padding='same', name='conv3D_{}'.format(blockid)))
    return model

def UpDoubleConv3DBlock(model, numfilters, blockid):
    model.add(UpSampling3D(size=(2,2,2), name='pool_{}'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}a'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}b'.format(blockid)))
    return model

def UpDoubleConv3DPadBlock(model, numfilters, blockid):
    model.add(ZeroPadding3D(padding=((0,0),(0,1),(0,1)), name='zeropad_{}'.format(blockid)))
    model.add(UpSampling3D(size=(2,2,2), name='pool_{}'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}a'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}b'.format(blockid)))
    return model

def UpDoubleConv3DBlock(model, numfilters, blockid):
    model.add(UpSampling3D(size=(2,2,2), name='pool_{}'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}a'.format(blockid)))
    model.add(Conv3D(numfilters,(3,3,3), activation='relu', padding='same', name='conv3D_{}b'.format(blockid)))
    return model
    
def build_3d_encoder_generator_model(shape=16):
    numfilters = 8
    model = Sequential()
    input_shape = (shape,shape,shape,1)
    model = Conv3DInputBlock(model, numfilters, input_shape)
    model = Conv3DBlock(model, 2*numfilters, 2)
    model = DoubleConv3DBlock(model, 4*numfilters, 3)
    model = DoubleConv3DBlock(model, 8*numfilters, 4)
    model = UpDoubleConv3DBlock(model, 8*numfilters, 5)
    model = UpDoubleConv3DBlock(model, 4*numfilters, 6)
    model = UpConv3DBlock(model, 2*numfilters, 7)
    model = UpConv3DBlock(model, numfilters, 8)
    model = LastConv3DBlock(model, 1, 9)
    print(model.summary())
    return model

def train_model(x_train, y_train, shape):
    model = build_3d_encoder_generator_model(16)
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    model.fit(x_train, y_train, epochs=5, batch_size=1)
    model.save('..{0}data{0}model.h5'.format(os.sep))
    return model