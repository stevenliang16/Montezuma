import random
import numpy as np
import utils

# Default architectures for the lower level controller/actor
defaultNSample = 256
defaultGamma = 0.99
defaultEpsilon = 1.0
defaultControllerEpsilon = [1.0]*6
defaultTau = 0.001

defaultAnnealSteps = 500000
defaultEndEpsilon = 0.1
defaultRandomPlaySteps = 100000

############
defaultMetaEpsilon = 1
defaultMetaNSamples = 256
controllerMemCap = 1000000
maxReward = 1
minReward = -1
trueSubgoalOrder = [2, 4, 3, 5]

class Agent:

    def __init__(self, net, actionSet, goalSet, metaEpsilon=defaultMetaEpsilon, epsilon=defaultEpsilon,
                 controllerEpsilon=defaultControllerEpsilon, tau=defaultTau):
        self.actionSet = actionSet
        self.controllerEpsilon = controllerEpsilon
        self.goalSet = goalSet
        self.metaEpsilon = metaEpsilon
        self.nSamples = defaultNSample #############
        self.metaNSamples = defaultMetaNSamples ################
        self.gamma = defaultGamma
        self.targetTau = tau
        self.net = net
        self.memory = []
        self.metaMemory = []

    def selectMove(self, state, goal):
        goalVec = utils.oneHot(goal)
        if self.controllerEpsilon[goal] < random.random():
            # predict action
            return np.argmax(self.net.controllerNet.predict([np.reshape(state, (1, 84, 84, 4)), np.asarray([goalVec])], verbose=0))
        return random.choice(self.actionSet)

    def setControllerEpsilon(self, epsilonArr):
        self.controllerEpsilon = epsilonArr

    def selectGoal(self, state):
        if self.metaEpsilon < random.random():
            # predict action
            pred = self.net.metaNet.predict(np.reshape(state, (1, 84, 84, 4)), verbose=0)
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
        if not useSparseReward:
            if action == 0:
                reward -= 0.1
            if die:
                reward -= 200.0
            # reward += distanceReward
        reward = np.minimum(reward, maxReward)
        reward = np.maximum(reward, minReward)
        return reward

    def store(self, experience, meta=False):
        if meta:
            self.metaMemory.append(experience)
            if len(self.metaMemory) > 50000:
                self.metaMemory = self.metaMemory[-50000:]
        else:
            self.memory.append(experience)
            if len(self.memory) > controllerMemCap:
                self.memory = self.memory[-controllerMemCap:]

    
    def _update(self, stepCount):
        exps = [random.choice(self.memory) for _ in range(self.nSamples)]
        # stateVectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=1) for exp in exps]))
        stateVector = []
        goalVector = []
        for exp in exps:
            stateVector.append(exp.state)
            goalVector.append(utils.oneHot(exp.goal))
        stateVector = np.asarray(stateVector)
        goalVector = np.asarray(goalVector)
        # nextStateVectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, exp.goal], axis=1) for exp in exps]))
        nextStateVector = []
        for exp in exps:
            nextStateVector.append(exp.next_state)
        nextStateVector = np.asarray(nextStateVector)
        rewardVectors = self.net.controllerNet.predict([stateVector, goalVector], verbose=0)
        nextStateRewardVectors = self.net.targetControllerNet.predict([nextStateVector, goalVector], verbose=0)

        for i, exp in enumerate(exps):
            rewardVectors[i][exp.action] = exp.reward
            if not exp.done:
                rewardVectors[i][exp.action] += self.gamma * max(nextStateRewardVectors[i])
        rewardVectors = np.asarray(rewardVectors)
        self.net.controllerNet.train_on_batch([stateVector, goalVector], rewardVectors)
        
        #Update target network
        controllerWeights = self.net.controllerNet.get_weights()
        controllerTargetWeights = self.net.targetControllerNet.get_weights()
        for i in range(len(controllerWeights)):
            controllerTargetWeights[i] = self.targetTau * controllerWeights[i] + (1 - self.targetTau) * controllerTargetWeights[i]
        self.net.targetControllerNet.set_weights(controllerTargetWeights)

    def _update_meta(self, stepCount):
        if 0 < len(self.metaMemory):
            exps = [random.choice(self.metaMemory) for _ in range(self.metaNSamples)]
            stateVectors = np.asarray([exp.state for exp in exps])
            nextStateVectors = np.asarray([exp.next_state for exp in exps])
            
            rewardVectors = self.net.metaNet.predict(stateVectors, verbose=0)
            nextStateRewardVectors = self.net.targetMetaNet.predict(nextStateVectors, verbose=0)

            for i, exp in enumerate(exps):
                rewardVectors[i][np.argmax(exp.goal)] = exp.reward
                if not exp.done:
                    rewardVectors[i][np.argmax(exp.goal)] += self.gamma * max(nextStateRewardVectors[i])
            self.net.metaNet.train_on_batch(stateVectors, rewardVectors)
            
            #Update target network
            metaWeights = self.net.metaNet.get_weights()
            metaTargetWeights = self.net.targetMetaNet.get_weights()
            for i in range(len(metaWeights)):
                metaTargetWeights[i] = self.targetTau * metaWeights[i] + (1 - self.targetTau) * metaTargetWeights[i]
            self.net.targetMetaNet.set_weights(metaTargetWeights)

    def update(self, stepCount, meta=False):
        if meta:
            self._update_meta(stepCount)
        else:
            self._update(stepCount)

    def annealMetaEpsilon(self, stepCount):
        self.metaEpsilon = defaultEndEpsilon + max(0, (defaultMetaEpsilon - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps)

    def annealControllerEpsilon(self, stepCount, goal):
        self.controllerEpsilon[goal] = defaultEndEpsilon + max(0, (defaultControllerEpsilon[goal] - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps)

