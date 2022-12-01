from __future__ import print_function
from custom_model.utils import *
import random 
import numpy as np

import tensorflow as tf
from keras import activations, initializers, constraints, regularizers
import keras.backend as K
from keras.layers import Layer, Reshape, Conv2D, BatchNormalization, TimeDistributed
from keras.layers import Lambda, Activation, Concatenate, Dense, RNN, Dropout
from keras.layers import Bidirectional, LayerNormalization, AveragePooling2D
from keras import Model
from keras.layers import Input

class TemporalAttention(Layer):
    
    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
        })
        return config
    
    def compute_output_shape(self, input_shapes):
        return input_shapes

    def build(self, input_shapes):
        T = input_shapes[-3]
        N = input_shapes[-2]
        F = input_shapes[-1]
        
        self.W1 = self.add_weight(shape=(N, 1),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=None,
                                  constraint=None)
        
        self.W2 = self.add_weight(shape=(F, N),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=None,
                                  constraint=None)
        self.W3 = self.add_weight(shape=(F, 1),
                                  initializer=self.kernel_initializer,
                                  name='W3',
                                  regularizer=None,
                                  constraint=None)
        self.Ve = self.add_weight(shape=(T, T),
                                  initializer=self.kernel_initializer,
                                  name='Ve',
                                  regularizer=None,
                                  constraint=None)
        self.be = self.add_weight(shape=(T, T),
                                  initializer=self.bias_initializer,
                                  name='be',
                                  regularizer=None,
                                  constraint=None)
        
        self.built = True

    def call(self, inputs, mask=None):
        x = K.permute_dimensions(inputs, (0,1,3,2)) #(b,T,F,N)
        r1 = K.dot(x,self.W1)
        lhs = K.dot(r1[...,0], self.W2) #(b,T,N)
        r2 = K.dot(inputs,self.W3) #(b,T,N)
        rhs = K.permute_dimensions(r2[...,0], (0,2,1)) #(b,N,T)
        product = K.batch_dot(lhs, rhs) #(b,T,T)
        E = tf.einsum('jk,ikl->ijl', self.Ve, tf.nn.sigmoid(product+self.be))
        kernel = tf.nn.softmax(E,axis=-2)
        conv = tf.einsum('ijkl,ilm->ijkm', K.permute_dimensions(inputs, (0,2,3,1)), kernel)
        
        return K.permute_dimensions(conv, (0,3,1,2))
    
    
class CrossAttention(Layer):
    def __init__(self,para,activation=None,
                 **kwargs):
        
        self.units = para['gc_units']
        self.nb = para['nb_classes']
        self.para = para
        self.pred = para['pred']
        self.nb_nodes = para['nb_nodes']
        self.activation = activations.get(activation)
        
        coordinates = np.arange(0, 130+65/(self.nb-1), 130/(self.nb-1))/130
        #coordinates = z_score(coordinates, para['stats'][0], para['stats'][2])
        coordinates = np.tile(coordinates.reshape((-1,1)), (self.nb_nodes,1,1))
        #assert len(coordinates)==self.nb
        self.mesh = K.constant(coordinates)

        super(CrossAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        cshapes = (1,)
        
        self.qlayer = Dense(self.units)#, depthwise=False
        self.qlayer.build(input_shape)
        w1 = self.qlayer.trainable_weights
        
        self.W = self.add_weight(shape=(self.nb_nodes, 1, self.units),
                                  initializer='glorot_uniform',
                                  name='kernel',
                                  regularizer=None,
                                  constraint=None)
        
        self.Wr = self.add_weight(shape=(self.nb_nodes, self.units, self.units),
                                  initializer='glorot_uniform',
                                  name='kernel1',
                                  regularizer=None,
                                  constraint=None)
        
        self.bias = self.add_weight(shape=(self.nb_nodes, self.units),
                                  initializer='zeros',
                                  name='bias',
                                  regularizer=None,
                                  constraint=None)
        
        self.biasr = self.add_weight(shape=(self.nb_nodes, self.units),
                                  initializer='zeros',
                                  name='biasr',
                                  regularizer=None,
                                  constraint=None)
        
        self.bias1 = self.add_weight(shape=(self.nb_nodes, self.nb),
                                  initializer='zeros',
                                  name='bias1',
                                  regularizer=None,
                                  constraint=None)
        

        w = w1+[self.W, self.Wr, self.bias, self.biasr, self.bias1]
        
        self._trainable_weights = w
        self.built = True

    def call(self, inputs):
        q = self.qlayer(inputs)
        k = K.batch_dot(self.mesh, self.W, axes=[-1,-2]) #(n,c,f)
        k = tf.transpose(k, perm=[1,0,2])+self.bias #(C,n,f)
        k = tf.transpose(k, perm=[1,0,2]) #(n,c,f)
        k = K.relu(k)
        
        k = K.batch_dot(k, self.Wr, axes=[-1,-2]) #(n,c,fr)
        k = tf.transpose(k, perm=[1,0,2])+self.biasr #(C,n,fr)

        k = tf.transpose(k, perm=[1,2,0]) #(b,T,N,F) x (N,F,C) -->(b,T,N,C)
        
        output = tf.einsum('btnf,nfc->btnc', q, k)+self.bias1
        
        return self.activation(output)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], input_shapes[2], self.nb)

class DynamicGC(Layer):
    def __init__(self, para, A_,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        
        self.k = para['adjacency_range']
        self.units = para['gc_units']

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)
        
        self.adj = K.constant(A_)

        super(DynamicGC, self).__init__(**kwargs)

    def build(self, input_shape):
        F = input_shape[-1]
        N = input_shape[-2]
        T = input_shape[-3]
        
        self.W1 = self.add_weight(shape=(N, N),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=None,
                                  constraint=None)
        
        self.W2 = self.add_weight(shape=(F, N),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=None,
                                  constraint=None)
        self.W3 = self.add_weight(shape=(F, self.units),
                                  initializer=self.kernel_initializer,
                                  name='W3',
                                  regularizer=None,
                                  constraint=None)
        self.bias1 = self.add_weight(shape=(N, ),
                                  initializer=self.kernel_initializer,
                                  name='b1',
                                  regularizer=None,
                                  constraint=None)
        self.bias2 = self.add_weight(shape=(self.units, ),
                                  initializer=self.bias_initializer,
                                  name='b2',
                                  regularizer=None,
                                  constraint=None)


        self.built = True

    def call(self, inputs):
        A = self.adj  # Adjacency matrix (N x N)
            
        feature = K.dot(K.permute_dimensions(inputs, (0,2,1)), A*self.W1)
        dense = K.dot(K.permute_dimensions(feature, (0,2,1)), self.W2)
        dense = K.bias_add(dense, self.bias1)
                
        mask = dense + -10e15 * (1.0 - A)
        mask = K.softmax(mask)

        node_features = tf.einsum('ijk,ikm->ijm', mask, inputs)  # (N x F)
        trans = K.dot(node_features, self.W3)

        out = K.bias_add(trans, self.bias2)


        output = self.activation(out)
        return output

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], self.units)
    
class SpatialAttention(Layer):
    def __init__(self,
                 para,
                 A_,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        
        self.units = para['gc_units']
        self.para = para
        self.adj = K.constant(A_)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation = activations.get(activation)


        super(SpatialAttention, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'para': self.para,
            'A_': self.adj,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'activation': self.activation,
        })
        return config
    
    def build(self, input_shape):
        F = input_shape[-1]
        N = input_shape[-2]
        T = input_shape[-3]
        
        self.W1 = self.add_weight(shape=(T, 1),
                                  initializer=self.kernel_initializer,
                                  name='W1',
                                  regularizer=None,
                                  constraint=None)
        
        self.W2 = self.add_weight(shape=(F, T),
                                  initializer=self.kernel_initializer,
                                  name='W2',
                                  regularizer=None,
                                  constraint=None)
        self.W3 = self.add_weight(shape=(F, 1),
                                  initializer=self.kernel_initializer,
                                  name='W3',
                                  regularizer=None,
                                  constraint=None)
        self.Ve = self.add_weight(shape=(N, N),
                                  initializer=self.kernel_initializer,
                                  name='Ve',
                                  regularizer=None,
                                  constraint=None)
        self.be = self.add_weight(shape=(N, N),
                                  initializer=self.kernel_initializer,
                                  name='be',
                                  regularizer=None,
                                  constraint=None)
        self.kernel = self.add_weight(shape=(F, self.units),
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=None,
                                  constraint=None)
        self.bias = self.add_weight(shape=(self.units,),
                                  initializer=self.kernel_initializer,
                                  name='bias',
                                  regularizer=None,
                                  constraint=None)

        self.built = True

    def call(self, inputs):
        A = self.adj # Adjacency matrix (N x N)
            
        x = K.permute_dimensions(inputs, (0,2,3,1)) #(b,T,F,N)
        r1 = K.dot(x,self.W1)
        lhs = K.dot(r1[...,0], self.W2) #(b,N,T)
        r2 = K.dot(inputs,self.W3) #(b,T,N)
        rhs = r2[...,0]
        product = K.batch_dot(lhs, rhs) #(b,N,N)
        E = tf.einsum('jk,ikm->ijm', self.Ve, tf.nn.sigmoid(product+self.be)) #(b,N,N)
        mask = E + -10e15 * (1.0 - A)
        mask = K.softmax(mask)
        #mask = tf.nn.softplus(mask)
        
        conv = tf.einsum('ijk,ilkm->iljm', mask, inputs)
        p = K.dot(conv, self.kernel)+self.bias
        
        return self.activation(p)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[-2], input_shapes[-1], self.units)
    
class NormalGC(Layer):
    def __init__(self, para, A_,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NormalGC, self).__init__(**kwargs)
        
        self.adj = K.constant(A_)
        self.units = para['gc_units']
        
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], self.units)

    def build(self, input_shapes):
        nb_nodes = input_shapes[-2]
        input_dim = input_shapes[-1]
        
        self.kernel = self.add_weight(shape=(nb_nodes, nb_nodes),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        self.kernel2 = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel2',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = K.dot(inputs, self.kernel2)
        W = self.adj*self.kernel
        X = K.dot(K.permute_dimensions(features, (0,2,1)), W)
        outputs = K.permute_dimensions(X, (0,2,1))
        if self.use_bias:
            outputs += self.bias
        return self.activation(outputs)

class STBlock(Layer):
    def __init__(self,para,
                 **kwargs):
        
        self.k = para['adjacency_range']
        self.units = para['gc_units']
        self.time_length = para['time_length']
        self.mode = para['mode']
        self.para = para
        
        A_ = directed_adj()
        A_ = np.linalg.matrix_power(A_, self.k)
        A_[A_>1]=1.
        
        self.adj = A_

        super(STBlock, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'para': self.para,
        })
        return config
    
    def build(self, input_shape):
        shapes = (input_shape[-3], input_shape[-2], self.units)
        
        self.ta = TemporalAttention()#, depthwise=False
        self.ta.build(shapes)
        w1 = self.ta.trainable_weights
        
        if self.mode=='dgc':
            self.sa = TimeDistributed(DynamicGC(self.para, self.adj))#, depthwise=False
            self.sa.build(input_shape)
            w2 = self.sa.trainable_weights
        if self.mode=='normalgc':
            self.sa = TimeDistributed(NormalGC(self.para, self.adj))#, depthwise=False
            self.sa.build(input_shape)
            w2 = self.sa.trainable_weights
        if self.mode=='spatialattention':
            self.sa = SpatialAttention(self.para, self.adj)#, depthwise=False
            self.sa.build(input_shape)
            w2 = self.sa.trainable_weights
        
        self.tconv = Conv2D(self.units, (self.time_length, 1), padding='same')#, depthwise=False
        self.tconv.build(shapes)
        w3 = self.tconv.trainable_weights
        
        self.rconv = Conv2D(self.units, (self.time_length, 1), padding='same')#, depthwise=False
        self.rconv.build(input_shape)
        w4 = self.rconv.trainable_weights

        w = w1+w2+w3+w4
        
        self._trainable_weights = w
        self.built = True

    def call(self, inputs):
        x = self.sa(inputs)
        x = self.ta(x)
        x = self.tconv(x)
        
        res = self.rconv(inputs)
        
        output = tf.nn.softplus(x)
        
        return output+res

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], input_shapes[1], input_shapes[2], self.units)