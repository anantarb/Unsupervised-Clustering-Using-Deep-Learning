import random
import numpy as np
import sys
from helper import *
from auto_en import Autoencoder
from keras.datasets import mnist
import matplotlib.pyplot as plt

if __name__ == '__main__':

    if sys.argv[2] == 'minst':
        if sys.argv[1] == 'conv':
            x_train, y_train, x_test, y_test = process_minst_dataset(model='conv')
            input_shape = x_train[1].shape
            ae = Autoencoder(input_shape)
            autoencoder, encoder = ae.make_conv_autoen_minst()
        else:
            x_train, y_train, x_test, y_test = process_minst_dataset()
            input_shape = x_train[1].shape
            ae = Autoencoder(input_shape)
            autoencoder, encoder = ae.make_autoen_minst()

    elif sys.argv[2] == 'cifar':
        if sys.argv[1] == 'conv':
            x_train, y_train, x_test, y_test = process_cifar_dataset(model='conv')
            input_shape = x_train[1].shape
            ae = Autoencoder(input_shape)
            autoencoder, encoder = ae.make_conv_autoen_cifar()
        else:
            x_train, y_train, x_test, y_test = process_cifar_dataset()
            input_shape = x_train[1].shape
            ae = Autoencoder(input_shape)
            autoencoder, encoder = ae.make_autoen_cifar()

    autoencoder.summary()
    ae.compile_model(autoencoder)
    ae.train_model(autoencoder, x_train, x_test)

    encoded_data = ae.eval_model(encoder, x_train)

    tsne(encoded_data, y_train, 2)

    print(v_score_using_kmeans(encoded_data, y_train, 10))
    print(v_score_using_GMM(encoded_data, y_train, 10))

    print(v_score_using_kmeans(x_train, y_train, 10))
    print(v_score_using_GMM(x_train, y_train, 10))
    
    x = []
    y = []
    for i in range(1, 50):
        x.append(i)
        y.append(v_score_using_GMM(encoded_data, y_train, i))

    a = [b * 100 for b in y]
    plt.plot(x, a)
    plt.title('Number of clusters vs V-Measure')
    plt.ylabel('V-Measure')
    plt.xlabel('Number of clusters')
    plt.show()

    encoded_data = ae.eval_model(encoder, x_test)

    print(v_score_using_kmeans(encoded_data, y_test, 10))
    print(v_score_using_GMM(encoded_data, y_test, 10))

    print(v_score_using_kmeans(x_test, y_test, 10))
    print(v_score_using_GMM(x_test, y_test, 10))