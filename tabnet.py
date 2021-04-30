# tabnet implementation from
# https://www.kaggle.com/marcusgawronsky/tabnet-in-tensorflow-2-0
# fixed decoder's loss function and masking process during training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

@tf.function
def identity(x):
    return x

@tf.function
def dummy_loss(y, t):
    return 0.


class GLUBlock(tf.keras.layers.Layer):
    def __init__(self, units=None, virtual_batch_size=128, momentum=0.02):
        super(GLUBlock, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
            
        self.fc_outout = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn_outout = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 
                                                            momentum=self.momentum)
        
        self.fc_gate = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn_gate = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 
                                                          momentum=self.momentum)
        
    def call(self, inputs, training=None):
        output = self.bn_outout(self.fc_outout(inputs), training=training)
        gate = self.bn_gate(self.fc_gate(inputs), training=training)
    
        return output * tf.keras.activations.sigmoid(gate) # GLU
    
    
class FeatureTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, units=None, virtual_batch_size=128, momentum=0.02, skip=False):
        super(FeatureTransformerBlock, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.skip = skip
        
    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
        
        self.initial = GLUBlock(units=self.units, 
                                virtual_batch_size=self.virtual_batch_size, 
                                momentum=self.momentum)
        self.residual =  GLUBlock(units=self.units, 
                                  virtual_batch_size=self.virtual_batch_size, 
                                  momentum=self.momentum)
        
    def call(self, inputs, training=None):
        initial = self.initial(inputs, training=training)
        
        if self.skip == True:
            initial += inputs

        residual = self.residual(initial, training=training) # skip
        
        return (initial + residual) * np.sqrt(0.5)
    
    
class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, units=None, virtual_batch_size=128, momentum=0.02):
        super(AttentiveTransformer, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
            
        self.fc = tf.keras.layers.Dense(self.units, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 
                                                     momentum=self.momentum)
        
    def call(self, inputs, priors=None, training=None):
        feature = self.bn(self.fc(inputs), training=training)
        if priors is None:
            output = feature
        else:
            output = feature * priors
        
        return tfa.activations.sparsemax(output)
    
    
class TabNetStep(tf.keras.layers.Layer):
    def __init__(self, units=None, virtual_batch_size=128, momentum=0.02):
        super(TabNetStep, self).__init__()
        self.units = units
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]
        
        self.unique = FeatureTransformerBlock(units=self.units, 
                                              virtual_batch_size=self.virtual_batch_size, 
                                              momentum=self.momentum,
                                              skip=True)
        self.attention = AttentiveTransformer(units=input_shape[-1], 
                                              virtual_batch_size=self.virtual_batch_size, 
                                              momentum=self.momentum)
        
    def call(self, inputs, shared, priors, training=None):  
        split = self.unique(shared, training=training)
        keys = self.attention(split, priors, training=training)
        masked = keys * inputs
        
        return split, masked, keys
    
    
class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(self, units: int =1, 
                 n_steps: int = 3, 
                 n_features: int = 8,
                 outputs: int = 1, 
                 gamma: float = 1.3,
                 epsilon: float = 1e-8, 
                 sparsity: float = 1e-5, 
                 virtual_batch_size=128, 
                 momentum=0.02):
        super(TabNetEncoder, self).__init__()
        
        self.units = units
        self.n_steps = n_steps
        self.n_features = n_features
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity
        
    def build(self, input_shape):            
        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 
                                                     momentum=self.momentum)
        self.shared_block = FeatureTransformerBlock(units = self.n_features, 
                                                    virtual_batch_size=self.virtual_batch_size, 
                                                    momentum=self.momentum)        
        self.initial_step = TabNetStep(units = self.n_features, 
                                       virtual_batch_size=self.virtual_batch_size, 
                                       momentum=self.momentum)
        self.steps = [TabNetStep(units = self.n_features, 
                                 virtual_batch_size=self.virtual_batch_size, 
                                 momentum=self.momentum) for _ in range(self.n_steps)]
        self.final = tf.keras.layers.Dense(units = self.units, 
                                           use_bias=False)
    

    def call(self, X, training=None):        
        entropy_loss = 0.
        encoded = 0.
        output = 0.
        importance = 0.
        prior = tf.reduce_mean(tf.ones_like(X), axis=0)
        
        B = prior * self.bn(X, training=training)
        shared = self.shared_block(B, training=training)
        _, masked, keys = self.initial_step(B, shared, prior, training=training)

        for step in self.steps:
            entropy_loss += tf.reduce_mean(tf.reduce_sum(-keys * tf.math.log(keys + self.epsilon), axis=-1)) / tf.cast(self.n_steps, tf.float32)
            prior *= (self.gamma - tf.reduce_mean(keys, axis=0))
            importance += keys
            
            shared = self.shared_block(masked, training=training)
            split, masked, keys = step(B, shared, prior, training=training)
            features = tf.keras.activations.relu(split)
            
            output += features
            encoded += split
            
        self.add_loss(self.sparsity * entropy_loss)
          
        prediction = self.final(output)
        return prediction, encoded, importance
    

class TabNetClassifier(tf.keras.Model):
    def __init__(self, outputs: int = 1, 
                 n_steps: int = 3, 
                 n_features: int = 8,
                 gamma: float = 1.3, 
                 epsilon: float = 1e-8, 
                 sparsity: float = 1e-5, 
                 feature_column=None, 
                 pretrained_encoder=None,
                 virtual_batch_size=128, 
                 momentum=0.02):
        super(TabNetClassifier, self).__init__()
        
        self.outputs = outputs
        self.n_steps = n_steps
        self.n_features = n_features
        self.feature_column = feature_column
        self.pretrained_encoder = pretrained_encoder
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity
        
        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column
            
        if pretrained_encoder is None:
            self.encoder = TabNetEncoder(units=outputs, 
                                        n_steps=n_steps, 
                                        n_features = n_features,
                                        outputs=outputs, 
                                        gamma=gamma, 
                                        epsilon=epsilon, 
                                        sparsity=sparsity,
                                        virtual_batch_size=self.virtual_batch_size, 
                                        momentum=momentum)
        else:
            self.encoder = pretrained_encoder

    def forward(self, X, training=None):
        X = self.feature(X)
        output, encoded, importance = self.encoder(X)
          
        prediction = tf.keras.activations.sigmoid(output)
        return prediction, encoded, importance
    
    def call(self, X, training=None):
        prediction, _, _ = self.forward(X)
        return prediction
    
    def transform(self, X, training=None):
        _, encoded, _ = self.forward(X)
        return encoded
    
    def explain(self, X, training=None):
        _, _, importance = self.forward(X)
        return importance    
    
    
class TabNetDecoder(tf.keras.layers.Layer):
    def __init__(self, units=1, 
                 n_steps = 3, 
                 n_features = 8,
                 outputs = 1, 
                 gamma = 1.3,
                 epsilon = 1e-8, 
                 sparsity = 1e-5, 
                 virtual_batch_size=128, 
                 momentum=0.02):
        super(TabNetDecoder, self).__init__()
        
        self.units = units
        self.n_steps = n_steps
        self.n_features = n_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
    def build(self, input_shape):
        self.shared_block = FeatureTransformerBlock(units = self.n_features, 
                                                    virtual_batch_size=self.virtual_batch_size, 
                                                    momentum=self.momentum)
        self.steps = [FeatureTransformerBlock(units = self.n_features,
                                              virtual_batch_size=self.virtual_batch_size, 
                                              momentum=self.momentum) for _ in range(self.n_steps)]
        self.fc = [tf.keras.layers.Dense(units = self.units) for _ in range(self.n_steps)]
    

    def call(self, X, training=None):
        decoded = 0.
        
        for ftb, fc in zip(self.steps, self.fc):
            shared = self.shared_block(X, training=training)
            feature = ftb(shared, training=training)
            output = fc(feature)
            
            decoded += output
        return decoded
    
    
class TabNetAutoencoder(tf.keras.Model):
    def __init__(self, outputs: int = 1, 
                 inputs: int = 12,
                 n_steps: int  = 3, 
                 n_features: int  = 8,
                 gamma: float = 1.3, 
                 epsilon: float = 1e-8, 
                 sparsity: float = 1e-5, 
                 feature_column=None, 
                 virtual_batch_size=128, 
                 momentum=0.02):
        super(TabNetAutoencoder, self).__init__()
        
        self.outputs = outputs
        self.inputs = inputs
        self.n_steps = n_steps
        self.n_features = n_features
        self.feature_column = feature_column
        self.virtual_batch_size = virtual_batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.sparsity = sparsity
        
        if feature_column is None:
            self.feature = tf.keras.layers.Lambda(identity)
        else:
            self.feature = feature_column
            
        self.encoder = TabNetEncoder(units=outputs, 
                                    n_steps=n_steps, 
                                    n_features = n_features,
                                    outputs=outputs, 
                                    gamma=gamma, 
                                    epsilon=epsilon, 
                                    sparsity=sparsity,
                                    virtual_batch_size=self.virtual_batch_size, 
                                    momentum=momentum)
        
        self.decoder = TabNetDecoder(units=inputs, 
                                     n_steps=n_steps, 
                                     n_features = n_features,
                                     virtual_batch_size=self.virtual_batch_size, 
                                     momentum=momentum)
        
        self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, 
                                                     momentum=momentum)
        
        self.do = tf.keras.layers.Dropout(0.25)

    def forward(self, X, training=None):
        X = self.feature(X)
        X = self.bn(X)
        
        # training mask
        M = self.do(tf.ones_like(X), training=training)
        D = X*M
        
        #encoder
        output, encoded, importance = self.encoder(D)
        prediction = tf.keras.activations.sigmoid(output)        
        
        return prediction, encoded, importance, X, M
    
    def call(self, X, training=None):
        # encode
        prediction, encoded, _, X, M = self.forward(X)
        T = X * (1 - M)

        #decode
        reconstruction = self.decoder(encoded)
        
        #loss
        loss  = tf.reduce_mean(tf.where(M != 0., tf.square(T-reconstruction), tf.zeros_like(reconstruction)))
        
        self.add_loss(loss)
        
        return prediction
    
    def transform(self, X, training=None):
        _, encoded, _, _, _ = self.forward(X)
        return encoded
    
    def explain(self, X, training=None):
        _, _, importance, _, _ = self.forward(X)
        return importance