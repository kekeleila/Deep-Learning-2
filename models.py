"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Lambda
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop
from keras.backend import squeeze
from keras.layers import concatenate
from keras.layers import  Input
import keras.backend as bk
from  keras.layers import merge
from keras.layers.wrappers import TimeDistributed
from keras.layers import normalization
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import tensorflow as tf
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048, data_size = None):
        """
        `model` = one of:
            lstm
            lrcn
            mlp
            conv_3d
            c3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.data_size = data_size

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn()
        elif model == 'lrcn_new':
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn_new()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.conv_3d()
        elif model == 'c3d':
            print("Loading C3D")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.c3d()
        elif model == 'singleFrame':
            self.input_shape = (80, 80, 3)
            self.model = self.singleFrame()
        elif model == 'lateFusion':
            self.model = self.lateFusion()
        elif model == 'earlyFusion':
            self.input_shape = (800, 80, 3)
            self.model = self.earlyFusion()
        elif model == 'slowFusion':
            self.model = self.slowFusion()
        elif model == 'twoStreams':
            self.model = self.twoStreams()

        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model




    def singleFrame(self):
        model = Sequential()
        #model.add(core.Reshape((80, 80, 3), input_shape=self.input_shape))
        model.add(Conv2D(kernel_size=11, filters=96, strides=(3,3), padding='same', activation='relu', input_shape=self.input_shape))
        model.add(normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
        model.add(Conv2D(kernel_size=5, filters=256, strides=(1,1), padding='same', activation='relu'))
        model.add(normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
        model.add(Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def lateFusion(self):
        input = Input(shape=(2,80,80,3))
        input1 = Lambda(lambda x: x[:,0,:,:,:], output_shape=(80, 80, 3))(input)
        input2 = Lambda(lambda x: x[:,1,:,:,:], output_shape=(80, 80, 3))(input)

        x = Conv2D(kernel_size=11, filters=96, strides=(3, 3), padding='same', activation='relu')(input1)
        x = normalization.BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2, strides=(2, 2))(x)
        x = (Conv2D(kernel_size=5, filters=256, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (normalization.BatchNormalization())(x)
        x = (MaxPooling2D(pool_size=2, strides=(2, 2)))(x)
        x = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (MaxPooling2D(pool_size=2, strides=(2, 2)))(x)

        y = Conv2D(kernel_size=11, filters=96, strides=(3, 3), padding='same', activation='relu')(input2)
        y = normalization.BatchNormalization()(y)
        y = MaxPooling2D(pool_size=2, strides=(2, 2))(y)
        y = (Conv2D(kernel_size=5, filters=256, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (normalization.BatchNormalization())(y)
        y = (MaxPooling2D(pool_size=2, strides=(2, 2)))(y)
        y = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (MaxPooling2D(pool_size=2, strides=(2, 2)))(y)


        merge = concatenate([x, y])
        merge = Flatten()(merge)
        merge = (Dense(4096, activation='relu'))(merge)
        merge = (Dense(4096, activation='relu'))(merge)
        merge = (Dense(self.nb_classes, activation='softmax'))(merge)
        model = Model(inputs=input, outputs= merge)


        return model

    def earlyFusion(self):
        model = Sequential()
        model.add(Conv2D(kernel_size=11, filters=96, strides=(3,3), padding='same', activation='relu', input_shape=self.input_shape))
        model.add(normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
        model.add(Conv2D(kernel_size=5, filters=256, strides=(1,1), padding='same', activation='relu'))
        model.add(normalization.BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, strides=(2,2)))
        model.add(Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model

    def slowFusion(self):

        input = Input(shape=(4, 320, 80,3))
        firstOutputList = []

        for i in range(4):
            cur = Lambda(lambda x: x[:, i, :, :, :], output_shape=(320, 80, 3))(input)
            x = Conv2D(kernel_size=11, filters = 48, strides=(3, 3), padding='same', activation='relu')(cur)
            x = normalization.BatchNormalization()(x)
            x = MaxPooling2D(pool_size=2, strides=(2, 2))(x)
            firstOutputList.append(x)

        newInput1 = concatenate([firstOutputList[0], firstOutputList[1]])
        newInput2 = concatenate([firstOutputList[2], firstOutputList[3]])
        output1 = Conv2D(kernel_size=11, filters=48, strides=(3, 3), padding='same', activation='relu')(newInput1)
        output1 = normalization.BatchNormalization()(output1)
        output1 = MaxPooling2D(pool_size=2, strides=(2, 2))(output1)
        output2 = Conv2D(kernel_size=11, filters=48, strides=(3, 3), padding='same', activation='relu')(newInput2)
        output2 = normalization.BatchNormalization()(output2)
        output2 = MaxPooling2D(pool_size=2, strides=(2, 2))(output2)
        newInput = concatenate([output1, output2])
        newInput = (Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))(newInput)
        newInput = (Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))(newInput)
        newInput = (Conv2D(kernel_size=3, filters=128, strides=(1, 1), padding='same', activation='relu'))(newInput)
        newInput = (MaxPooling2D(pool_size=2, strides=(2, 2)))(newInput)

        output = Flatten()(newInput)
        output = (Dense(1024, activation='relu'))(output)
        output = (Dense(1024, activation='relu'))(output)
        output = (Dense(self.nb_classes, activation='softmax'))(output)
        model = Model(inputs=input, outputs= output)

        return model

    def twoStreams(self):
        input = Input(shape=(80, 80, 3))
        input1 = Lambda(lambda x: x[:, 20:60, 20:60, :], output_shape=(40, 40, 3))(input)
        input2 = Lambda(lambda x: tf.strided_slice(x, [0, 0, 0, 0], [self.data_size, -1, -1, 3], [1, 2, 2, 1]), output_shape=(40, 40, 3))(input)

        x = Conv2D(kernel_size=11, filters=96, strides=(3, 3), padding='same', activation='relu')(input1)
        x = normalization.BatchNormalization()(x)
        x = MaxPooling2D(pool_size=2, strides=(2, 2))(x)
        x = (Conv2D(kernel_size=5, filters=256, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (normalization.BatchNormalization())(x)
        x = (MaxPooling2D(pool_size=2, strides=(2, 2)))(x)
        x = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))(x)
        x = (MaxPooling2D(pool_size=2, strides=(2, 2)))(x)

        y = Conv2D(kernel_size=11, filters=96, strides=(3, 3), padding='same', activation='relu')(input2)
        y = normalization.BatchNormalization()(y)
        y = MaxPooling2D(pool_size=2, strides=(2, 2))(y)
        y = (Conv2D(kernel_size=5, filters=256, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (normalization.BatchNormalization())(y)
        y = (MaxPooling2D(pool_size=2, strides=(2, 2)))(y)
        y = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu'))(y)
        y = (MaxPooling2D(pool_size=2, strides=(2, 2)))(y)


        merge = concatenate([x, y])
        merge = Flatten()(merge)
        merge = (Dense(4096, activation='relu'))(merge)
        merge = (Dense(4096, activation='relu'))(merge)
        merge = (Dense(self.nb_classes, activation='softmax'))(merge)
        model = Model(inputs=input, outputs= merge)
        return model

    def lrcn_new(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        model = Sequential()
        model.add(TimeDistributed(Conv2D(kernel_size=11, filters=96, strides=(3,3), padding='same', activation='relu'), input_shape=self.input_shape))
        model.add(TimeDistributed(normalization.BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D(pool_size=2, strides=(2,2))))
        model.add(TimeDistributed(Conv2D(kernel_size=5, filters=256, strides=(1,1), padding='same', activation='relu')))
        model.add(TimeDistributed(normalization.BatchNormalization()))
        model.add(TimeDistributed(MaxPooling2D(pool_size=2, strides=(2,2))))
        model.add(TimeDistributed(Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(kernel_size=3, filters=384, strides=(1, 1), padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(kernel_size=3, filters=256, strides=(1, 1), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=2, strides=(2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(4096, activation='relu')))
        model.add(TimeDistributed(Dense(4096, activation='relu')))

        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model






