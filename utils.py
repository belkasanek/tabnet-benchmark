from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def cast_save_memmory(df, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS):
    df.loc[:, NUMERIC_COLUMNS] = df.loc[:, NUMERIC_COLUMNS].astype(np.float32)
    df.loc[:, CATEGORICAL_COLUMNS] = df.loc[:, CATEGORICAL_COLUMNS].astype(np.int8)
    df.loc[:, 'target'] = df.loc[:, 'target'].astype(np.int8)
    df.loc[:, 'id'] = df.loc[:, 'id'].astype(np.int32)    
    
def plot_metrics(history):
    metrics = ['loss', 'auc']
    plt.figure(figsize=(12, 9))
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color='blue', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.grid()
        plt.legend()

def make_ds(X, y, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, shuffle=True, batch_size=10000):
    ds = tf.data.Dataset.from_tensor_slices(({col: X[col] for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS},
                                             y)).cache()
    if shuffle:
        ds = ds.shuffle(len(X)).repeat()
    return ds

def create_dataset(X, y, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, shuffle=True, batch_size=10000):
    ds = tf.data.Dataset.from_tensor_slices(({col: X[col] for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS},
                                             y)).cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
        
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_feature(x, dimension=1):
    if x.dtype == np.float32:
        return tf.feature_column.numeric_column(x.name)
    else:
        return tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(x.name, num_buckets=x.max() + 1, default_value=0),
        dimension=dimension)