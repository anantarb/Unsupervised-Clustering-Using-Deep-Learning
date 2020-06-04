from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from sklearn.mixture import GaussianMixture
import pickle
from keras.datasets import mnist
import pandas as pd
import seaborn as sn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt



def v_score_using_kmeans(x, y, cluster):
    km = KMeans(n_clusters=cluster)
    pred = km.fit_predict(x)
    return v_measure_score(y, pred)


def v_score_using_GMM(x, y, cluster):
    gmm = GaussianMixture(n_components=cluster)
    pred = gmm.fit_predict(x)
    return v_measure_score(y, pred)

def tsne(x, y, components):
    model = TSNE(n_components=components, random_state=0)
    tsne_data = model.fit_transform(x[0:10000])
    tsne_data = np.vstack((tsne_data.T, y[0:10000])).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.show()

def process_cifar_dataset(model='dense'):
    with open("cifar-10-batches-py/data_batch_1", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    x_train = dict[b'data'].astype('float32') / 255.
    y_train = np.array(dict[b'labels'])

    with open("cifar-10-batches-py/data_batch_2", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    x_train = np.concatenate((x_train, dict[b'data'].astype('float32') / 255.), axis=0)
    y_train = np.concatenate((y_train, np.array(dict[b'labels'])))

    with open("cifar-10-batches-py/data_batch_3", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    x_train = np.concatenate((x_train, dict[b'data'].astype('float32') / 255.), axis=0)
    y_train = np.concatenate((y_train, np.array(dict[b'labels'])))

    with open("cifar-10-batches-py/data_batch_4", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    x_train = np.concatenate((x_train, dict[b'data'].astype('float32') / 255.), axis=0)
    y_train = np.concatenate((y_train, np.array(dict[b'labels'])))

    with open("cifar-10-batches-py/data_batch_5", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    x_train = np.concatenate((x_train, dict[b'data'].astype('float32') / 255.), axis=0)
    y_train = np.concatenate((y_train, np.array(dict[b'labels'])))

    with open("cifar-10-batches-py/test_batch", 'rb') as fo:
       dict = pickle.load(fo, encoding='bytes')
    x_test = dict[b'data'].astype('float32') / 255.
    y_test = np.array(dict[b'labels'])
    
    if model == 'conv':
        x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
        x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    return x_train, y_train, x_test, y_test

def process_minst_dataset(model='dense'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    if model == 'conv':
        x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    else:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train, x_test, y_test