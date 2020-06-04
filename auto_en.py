import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt

# main implmenetation of autoencoder
class Autoencoder():

    # input_shape is input shape of our training data
    def __init__(self, input_shape):
        self.inputshape = input_shape
    
    # autoencoder model
    def make_autoen_minst(self):
        inputs = Input(shape=(self.inputshape))
        encoded = Dense(500, activation='relu')(inputs)
        encoded = Dense(500, activation='relu')(encoded)
        encoded = Dense(2000, activation='relu')(encoded)
        encoded = Dense(10)(encoded)
    
        decoded = Dense(2000, activation='relu')(encoded)
        decoded = Dense(500, activation='relu')(decoded)
        decoded = Dense(500, activation='relu')(decoded)
        decoded = Dense(784)(decoded)
    
        autoencoder = Model(inputs, decoded)
        self.autoencoder = autoencoder
    
        encoder = Model(autoencoder.input, autoencoder.layers[-5].output)
        
        return autoencoder, encoder

    def make_autoen_cifar(self):
        inputs = Input(shape=(self.inputshape))
        encoded = Dense(500, activation='relu')(inputs)
        encoded = Dense(500, activation='relu')(encoded)
        encoded = Dense(2000, activation='relu')(encoded)
        encoded = Dense(35)(encoded)
    
        decoded = Dense(2000, activation='relu')(encoded)
        decoded = Dense(500, activation='relu')(decoded)
        decoded = Dense(500, activation='relu')(decoded)
        decoded = Dense(3072)(decoded)
    
        autoencoder = Model(inputs, decoded)
        self.autoencoder = autoencoder
    
        encoder = Model(autoencoder.input, autoencoder.layers[-5].output)
        
        return autoencoder, encoder

    def make_conv_autoen_minst(self):
        inputs = Input(shape=(28, 28, 1))
        encoded = Conv2D(32, 5, activation='relu', padding='same', strides=(2, 2))(inputs)
        encoded = Conv2D(64, 5, activation='relu', padding='same', strides=(2, 2))(encoded)
        encoded = Conv2D(128, 3, activation='relu', padding='valid', strides=(2, 2))(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(10)(encoded)
      
        decoded = Dense(1152, activation='relu')(encoded)
        decoded = Reshape((3, 3, 128))(decoded)
        decoded = Conv2DTranspose(64, 3, activation='relu', padding='valid', strides=(2, 2))(decoded)
        decoded = Conv2DTranspose(32, 5, activation='relu', padding='same', strides=(2, 2))(decoded)
        decoded = Conv2DTranspose(1, 5, padding='same', strides=(2, 2))(decoded)
    
        autoencoder = Model(inputs, decoded)
        self.autoencoder = autoencoder
    
        encoder = Model(autoencoder.input, autoencoder.layers[-6].output)
        
        return autoencoder, encoder

    def make_conv_autoen_cifar(self):
        inputs = Input(shape=(32, 32, 3))
        encoded = Conv2D(32, 5, activation='relu', padding='same', strides=(2, 2))(inputs)
        encoded = Conv2D(64, 5, activation='relu', padding='same', strides=(2, 2))(encoded)
        encoded = Conv2D(128, 4, activation='relu', padding='valid', strides=(2, 2))(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(35)(encoded)
      
        decoded = Dense(1152, activation='relu')(encoded)
        decoded = Reshape((3, 3, 128))(decoded)
        decoded = Conv2DTranspose(64, 4, activation='relu', padding='valid', strides=(2, 2))(decoded)
        decoded = Conv2DTranspose(32, 5, activation='relu', padding='same', strides=(2, 2))(decoded)
        decoded = Conv2DTranspose(3, 5, padding='same', strides=(2, 2))(decoded)
    
        autoencoder = Model(inputs, decoded)
        self.autoencoder = autoencoder
    
        encoder = Model(autoencoder.input, autoencoder.layers[-6].output)
        
        return autoencoder, encoder
    
    # method to compile the model
    def compile_model(self, model):
        model.compile(optimizer='adam', metrics=['mse', 'accuracy'], loss='mse')
    
    # trains the model
    def train_model(self, model, x_train, x_test):
        
        history = model.fit(x_train, x_train, epochs=50, shuffle=True, batch_size=256, verbose=2, validation_data=(x_test, x_test))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    # method to predict the data after model is trained
    def eval_model(self, model, data):
        preds = model.predict(data)
        return preds
