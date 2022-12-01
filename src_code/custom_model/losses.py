import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras import backend as K

def nll_beta():
    def inloss(y_true, y_pred):
        
        value = y_true
        w = y_pred[...,:1]
        k = 1/(y_pred[...,1:]**2+1e-4)+0.2
        w = w*0.98+0.01
        
        a = w*k+1
        b = (1-w)*k+1
        
        n1 = tfp.distributions.Beta(a, b)
        
        loss = n1.prob(value)
        summ = tf.math.log(loss+1e-10) * -1
        
        return summ
    return inloss


def focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        return loss*100
    return focal_loss_fixed