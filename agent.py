import random
import numpy as np
import utils

from memory import Memory

# Default architectures for the lower level controller/actor
defaultNSample = 256
defaultGamma = 0.99
defaultEpsilon = 1.0
defaultControllerEpsilon = [1.0]*4
defaultTau = 0.001

defaultAnnealSteps = 500000
defaultEndEpsilon = 0.1
defaultRandomPlaySteps = 200000

defaultMetaEpsilon = 1
defaultMetaNSamples = 256
controllerMemCap = 1000000
metaMemCap = 50000
maxReward = 1
minReward = -1
#trueSubgoalOrder = [0, 1, 2]
hardUpdateFrequency = 10000 * 4

class Agent:

    def __init__(self, net, actionSet, goalSet, metaEpsilon=defaultMetaEpsilon, epsilon=defaultEpsilon,
                 controllerEpsilon=defaultControllerEpsilon, tau=defaultTau):
        self.actionSet = actionSet
        self.controllerEpsilon = controllerEpsilon
        self.goalSet = goalSet
        self.metaEpsilon = metaEpsilon
        self.nSamples = defaultNSample 
        self.metaNSamples = defaultMetaNSamples 
        self.gamma = defaultGamma
        self.targetTau = tau
        self.net = net
        self.memory = Memory(controllerMemCap)
        self.metaMemory = Memory(metaMemCap)

    def selectMove(self, state, goal):
        goalVec = utils.oneHot(goal)
        if self.controllerEpsilon[goal] < random.random():
            # predict action
            dummyYtrue = np.zeros((1, 8))
            dummyMask = np.zeros((1, 8))
            return np.argmax(self.net.controllerNet.predict([np.reshape(state, (1, 84, 84, 4)), np.asarray([goalVec]), dummyYtrue, dummyMask], verbose=0)[1])
        return random.choice(self.actionSet)

    def setControllerEpsilon(self, epsilonArr):
        self.controllerEpsilon = epsilonArr

    def selectGoal(self, state):
        if self.metaEpsilon < random.random():
            # predict action
            pred = self.net.metaNet.predict([np.reshape(state, (1, 84, 84, 4)), np.zeros((1,3)), np.zeros((1,3))], verbose=0)[1]
            return np.argmax(pred)
        return random.choice(self.goalSet)

    def selectTrueGoal(self, goalNum):
        return trueSubgoalOrder[goalNum]

    def setMetaEpsilon(self, epsilon):
        self.metaEpsilon = epsilon

    def criticize(self, reachGoal, action, die, distanceReward, useSparseReward):
        reward = 0.0
        if reachGoal:
            reward += 50.0
        # if die:
        #     reward -= 200.0
        if not useSparseReward:
            # if action == 0:
            #     reward -= 0.1
            reward += distanceReward
        reward = np.minimum(reward, maxReward)
        reward = np.maximum(reward, minReward)
        return reward

    def store(self, experience, meta=False):
        if meta:
            self.metaMemory.add(np.abs(experience.reward), experience)
        else:
            self.memory.add(np.abs(experience.reward), experience)

    
    def _update(self, stepCount):
        batches = self.memory.sample(self.nSamples)
        stateVector = []
        goalVector = []
        for batch in batches:
            exp = batch[1]
            stateVector.append(exp.state)
            goalVector.append(utils.oneHot(exp.goal))
        stateVector = np.asarray(stateVector)
        goalVector = np.asarray(goalVector)
        nextStateVector = []
        for batch in batches:
            exp = batch[1]
            nextStateVector.append(exp.next_state)
        nextStateVector = np.asarray(nextStateVector)
        rewardVectors = self.net.controllerNet.predict([stateVector, goalVector, np.zeros((self.nSamples,8)), np.zeros((self.nSamples, 8 ))], verbose=0)[1]
        
        rewardVectorsCopy = np.copy(rewardVectors)
        rewardVectors = np.zeros((self.nSamples, 8))
        nextStateRewardVectors = self.net.targetControllerNet.predict([nextStateVector, goalVector, np.zeros((self.nSamples,8)), np.zeros((self.nSamples, 8 ))], verbose=0)[1]
        
        maskVector = np.zeros((self.nSamples, 8))
        for i, batch in enumerate(batches):
            exp = batch[1]
            idx = batch[0]
            maskVector[i, exp.action] = 1. 
            rewardVectors[i][exp.action] = exp.reward
            if not exp.done:
                rewardVectors[i][exp.action] += self.gamma * max(nextStateRewardVectors[i])
            self.memory.update(idx, np.abs(rewardVectors[i][exp.action] - rewardVectorsCopy[i][exp.action]))
        rewardVectors = np.asarray(rewardVectors)
        loss = self.net.controllerNet.train_on_batch([stateVector, goalVector, rewardVectors, maskVector], [np.zeros(self.nSamples), rewardVectors])
        #Update target network
        controllerWeights = self.net.controllerNet.get_weights()
        controllerTargetWeights = self.net.targetControllerNet.get_weights()
        for i in range(len(controllerWeights)):
            controllerTargetWeights[i] = self.targetTau * controllerWeights[i] + (1 - self.targetTau) * controllerTargetWeights[i]
        self.net.targetControllerNet.set_weights(controllerTargetWeights)
        return loss
        

    def _update_meta(self, stepCount):
        batches = self.metaMemory.sample(self.metaNSamples)
        stateVectors = np.asarray([batch[1].state for batch in batches])
        nextStateVectors = np.asarray([batch[1].next_state for batch in batches])
        
        rewardVectors = self.net.metaNet.predict([stateVectors, np.zeros((self.nSamples,3)), np.zeros((self.nSamples, 3))], verbose=0)[1]
        rewardVectorsCopy = np.copy(rewardVectors)
        rewardVectors = np.zeros((self.metaNSamples, 3))
        nextStateRewardVectors = self.net.targetMetaNet.predict([nextStateVectors, np.zeros((self.nSamples,3)), np.zeros((self.nSamples, 3))], verbose=0)[1]
        maskVector = np.zeros((self.metaNSamples, 3))
        
        for i, batch in enumerate(batches):
            exp = batch[1]
            idx = batch[0]
            maskVector[i, exp.goal] = 1. 
            rewardVectors[i][exp.goal] = exp.reward
            if not exp.done:
                rewardVectors[i][np.argmax(exp.goal)] += self.gamma * max(nextStateRewardVectors[i])
            self.metaMemory.update(idx, np.abs(rewardVectors[i][exp.goal] - rewardVectorsCopy[i][exp.goal]))
        loss = self.net.metaNet.train_on_batch([stateVectors, rewardVectors, maskVector], [np.zeros(self.nSamples), rewardVectors])
        
        #Update target network
        metaWeights = self.net.metaNet.get_weights()
        metaTargetWeights = self.net.targetMetaNet.get_weights()
        for i in range(len(metaWeights)):
            metaTargetWeights[i] = self.targetTau * metaWeights[i] + (1 - self.targetTau) * metaTargetWeights[i]
        self.net.targetMetaNet.set_weights(metaTargetWeights)
        return loss
    def update(self, stepCount, meta=False):
        if meta:
            loss = self._update_meta(stepCount)
        else:
            loss = self._update(stepCount)
        return loss

    def annealMetaEpsilon(self, stepCount):
        self.metaEpsilon = defaultEndEpsilon + max(0, (defaultMetaEpsilon - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps)

    def annealControllerEpsilon(self, stepCount, goal):
        self.controllerEpsilon[goal] = defaultEndEpsilon + max(0, (defaultControllerEpsilon[goal] - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps)
