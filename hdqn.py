from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate, Lambda
from keras import backend as K
from keras import optimizers
import numpy as np
actionSetSize = 8
goalSetSize = 3

HUBER_DELTA = 0.5
'''
def smoothL1(y_true, y_pred):
       x   = K.abs(y_true - y_pred)
       x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
       return  K.sum(x)
'''
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

    
def clipped_masked_error(args):
        y_true, y_pred, mask = args
        loss = huber_loss(y_true, y_pred, 1)
        loss *= mask  # apply element-wise mask
        return K.sum(loss, axis=-1)

    
class Hdqn:
    def __init__(self):
        metrics = [mean_q]
        # Refer https://keras.io/getting-started/functional-api-guide/ for creating complex non-sequencial net
        state = Input(shape=(84,84,4))
        goal = Input(shape=(3,))
        xState = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(state)
        xState = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xState)
        xState = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xState)
        outState = Flatten()(xState)
        mergedSignal = concatenate([outState, goal], axis=-1)
        mergedSignal = Dense(512, activation = 'relu')(mergedSignal)
        y_pred = Dense(actionSetSize, activation = 'relu')(mergedSignal)
        y_true = Input(name='y_true', shape=(actionSetSize,))
        mask = Input(name='mask', shape=(actionSetSize,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        originalInput = [state, goal]
        controller = Model(inputs = originalInput + [y_true, mask], outputs=[loss_out, y_pred])     
        combined_metrics = {controller.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        controller.compile(loss = losses, optimizer = rmsProp, metrics=combined_metrics)
        
        
        # Target network architecture
        state = Input(shape=(84,84,4))
        goal = Input(shape=(3,))
        xState = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(state)
        xState = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xState)
        xState = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xState)
        outState = Flatten()(xState)
        mergedSignal = concatenate([outState, goal], axis=-1)
        mergedSignal = Dense(512, activation = 'relu')(mergedSignal)
        y_pred = Dense(actionSetSize, activation = 'relu')(mergedSignal)
        y_true = Input(name='y_true', shape=(actionSetSize,))
        mask = Input(name='mask', shape=(actionSetSize,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        originalInput = [state, goal]
        controllerTarget = Model(inputs = originalInput + [y_true, mask], outputs=[loss_out, y_pred])   
        combined_metrics = {controllerTarget.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        controllerTarget.compile(loss = losses, optimizer = rmsProp, metrics=combined_metrics)
        
        
        '''
        stateTarget = Input(shape = (84,84,4))
        goalTarget = Input(shape = (3,))
        xStateTarget = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(stateTarget)
        xStateTarget = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xStateTarget)
        xStateTarget = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xStateTarget)
        outStateTarget = Flatten()(xStateTarget)
        mergedSignalTarget = concatenate([outStateTarget, goalTarget], axis=-1)
        mergedSignalTarget = Dense(512, activation = 'relu')(mergedSignalTarget)
        outputActionTarget = Dense(actionSetSize, activation = 'relu')(mergedSignalTarget)
        controllerTarget = Model(inputs=[stateTarget, goalTarget], outputs=outputActionTarget)
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        
        controllerTarget.compile(loss = smoothL1, optimizer = rmsProp)
        '''
        
        state = Input(shape=(84,84,4))
        xState = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(state)
        xState = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xState)
        xState = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xState)
        outState = Flatten()(xState)
        outState = Dense(512, activation = 'relu')(outState)
        y_pred = Dense(goalSetSize, activation = 'relu')(outState)
        y_true = Input(name='y_true', shape=(goalSetSize,))
        mask = Input(name='mask', shape=(goalSetSize,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        originalInput = [state]
        meta = Model(inputs = originalInput + [y_true, mask], outputs=[loss_out, y_pred])   
        combined_metrics = {meta.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        meta.compile(loss = losses, optimizer = rmsProp, metrics=combined_metrics)
        
        
        
        
        state = Input(shape=(84,84,4))
        xState = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(state)
        xState = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xState)
        xState = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xState)
        outState = Flatten()(xState)
        outState = Dense(512, activation = 'relu')(outState)
        y_pred = Dense(goalSetSize, activation = 'relu')(outState)
        y_true = Input(name='y_true', shape=(goalSetSize,))
        mask = Input(name='mask', shape=(goalSetSize,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        originalInput = [state]
        metaTarget = Model(inputs = originalInput + [y_true, mask], outputs=[loss_out, y_pred])   
        combined_metrics = {metaTarget.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        rmsProp = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=1e-08, decay=0.0)
        metaTarget.compile(loss = losses, optimizer = rmsProp, metrics=combined_metrics)
        
        self.controllerNet = controller
        self.metaNet = meta
        self.targetControllerNet = controllerTarget
        self.targetMetaNet = metaTarget
        
    def saveWeight(self, stepCount):
        self.controllerNet.save_weights('controllerNet_' + str(stepCount) + '.h5')
        self.metaNet.save_weights('metaNet_' + str(stepCount) + '.h5')

    def loadWeight(self):
        path = 'weight/'
        self.controllerNet = load_model(path + 'controllerNet.h5', custom_objects = {'huber_loss' :huber_loss})
        self.targetControllerNet = load_model(path + 'controllerNet.h5', custom_objects = {'huber_loss' :huber_loss})
        self.metaNet = load_model(path + 'metaNet.h5', custom_objects = {'huber_loss' :huber_loss})
        self.targetMetaNet = load_model(path + 'metaNet.h5', custom_objects = {'huber_loss' :huber_loss})
