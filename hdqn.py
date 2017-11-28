from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, Input, concatenate

actionSetSize = 8
goalSetSize = 4

class Hdqn:
    
    def __init__(self):
        
        # Refer https://keras.io/getting-started/functional-api-guide/ for creating complex non-sequencial net
        state = Input(shape=(84,84,4))
        goal = Input(shape=(4,))
        xState = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(state)
        xState = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xState)
        xState = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xState)
        outState = Flatten()(xState)
        mergedSignal = concatenate([outState, goal], axis=-1)
        mergedSignal = Dense(512, activation = 'relu')(mergedSignal)
        outputAction = Dense(actionSetSize, activation = 'relu')(mergedSignal)
        controller = Model(inputs=[state, goal], outputs=outputAction)
        controller.compile(loss = 'mean_squared_error', optimizer = 'Adadelta')
        
        
        # Target network architecture
        stateTarget = Input(shape = (84,84,4))
        goalTarget = Input(shape = (4,))
        xStateTarget = Conv2D(32, (8,8), strides = 4, activation = 'relu', padding = 'valid')(stateTarget)
        xStateTarget = Conv2D(64, (4,4), strides = 2, activation = 'relu', padding = 'valid')(xStateTarget)
        xStateTarget = Conv2D(64, (3,3), strides = 1, activation = 'relu', padding = 'valid')(xStateTarget)
        outStateTarget = Flatten()(xStateTarget)
        mergedSignalTarget = concatenate([outStateTarget, goalTarget], axis=-1)
        mergedSignalTarget = Dense(512, activation = 'relu')(mergedSignalTarget)
        outputActionTarget = Dense(actionSetSize, activation = 'relu')(mergedSignalTarget)
        controllerTarget = Model(inputs=[stateTarget, goalTarget], outputs=outputActionTarget)
        controllerTarget.compile(loss = 'mean_squared_error', optimizer = 'Adadelta')
        
        meta = Sequential()
        meta.add(Conv2D(32, (8, 8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84, 84, 4)))
        meta.add(Conv2D(64, (4, 4), strides = 2, activation = 'relu', padding = 'valid'))
        meta.add(Conv2D(64, (3, 3), strides = 1, activation = 'relu', padding = 'valid'))
        meta.add(Flatten())
        meta.add(Dense(512, activation = 'relu'))
        meta.add(Dense(goalSetSize, activation = 'relu')) # Total number of actions = 18 ?????
        meta.compile(loss = 'mean_squared_error', optimizer = 'Adadelta')

        metaTarget = Sequential()
        metaTarget.add(Conv2D(32, (8, 8), strides = 4, activation = 'relu', padding = 'valid', input_shape = (84, 84, 4)))
        metaTarget.add(Conv2D(64, (4, 4), strides = 2, activation = 'relu', padding = 'valid'))
        metaTarget.add(Conv2D(64, (3, 3), strides = 1, activation = 'relu', padding = 'valid'))
        metaTarget.add(Flatten())
        metaTarget.add(Dense(512, activation = 'relu'))
        metaTarget.add(Dense(goalSetSize, activation = 'relu')) # Total number of actions = 18 ?????
        metaTarget.compile(loss = 'mean_squared_error', optimizer = 'Adadelta')
        
        self.controllerNet = controller
        self.metaNet = meta
        self.targetControllerNet = controllerTarget
        self.targetMetaNet = metaTarget
        
    def saveWeight(self, stepCount):
        self.controllerNet.save('controllerNet_' + str(stepCount) + '.h5')
        self.metaNet.save('metaNet_' + str(stepCount) + '.h5')

    def loadWeight(self):
        path = 'weight/'
        self.controllerNet = load_model(path + 'controllerNet.h5')
        self.targetControllerNet = load_model(path + 'controllerNet.h5')
        self.metaNet = load_model(path + 'metaNet.h5')
        self.targetMetaNet = load_model(path + 'metaNet.h5')