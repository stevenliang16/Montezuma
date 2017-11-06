import random
import numpy as np
import utils

# Default architectures for the lower level controller/actor
defaultNSample = 1000
defaultGamma = 0.975
defaultEpsilon = 1.0
defaultControllerEpsilon = [1.0]*6
defaultTau = 0.001

defaultAnnealSteps = 50000
defaultEndEpsilon = 0.1
defaultRandomPlaySteps = 10000

############
defaultMetaEpsilon = 1
defaultMetaNSamples = 1000

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
            return np.argmax(self.net.controller.predict([state, goalVec], verbose=0))
        return random.choice(self.actionSet)

    def selectGoal(self, state):
        if self.metaEpsilon < random.random():
            # predict action
            pred = self.net.meta_controller.predict(state, verbose=0)
            print("pred shape: " + str(pred.shape))
            return np.argmax(pred)
        print("Exploring")
        return random.choice(self.goalSet)

    def criticize(self, goalReached):
        return 1.0 if goalReached else 0.0

    def store(self, experience, meta=False):
        if meta:
            self.metaMemory.append(experience)
            if len(self.metaMemory) > 1000000:
                # -100????
                self.metaMemory = self.metaMemory[-100:]
        else:
            self.memory.append(experience)
            if len(self.memory) > 1000000:
                self.memory = self.memory[-1000000:]

    
    def _update(self):
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
        self.net.controllerNet.fit([stateVector, goalVector], rewardVectors, verbose=0)
        
        #Update target network
        controllerWeights = self.net.controllerNet.get_weights()
        controllerTargetWeights = self.net.targetControllerNet.get_weights()
        for i in range(len(controllerWeights)):
            controllerTargetWeights[i] = self.targetTau * controllerWeights[i] + (1 - self.targetTau) * controllerTargetWeights[i]
        self.net.targetControllerNet.set_weights(controllerTargetWeights)

    def _update_meta(self):
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
            self.net.metaNet.fit(stateVectors, rewardVectors, verbose=0)
            
            #Update target network
            metaWeights = self.net.metaNet.get_weights()
            metaTargetWeights = self.net.targetMetaNet.get_weights()
            for i in range(len(metaWeights)):
                metaTargetWeights[i] = self.targetTau * metaWeights[i] + (1 - self.targetTau) * metaTargetWeights[i]
            self.net.targetMetaNet.set_weights(metaTargetWeights)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update()

    def annealMetaEpsilon(self, stepCount):
        self.metaEpsilon = defaultEndEpsilon + (defaultMetaEpsilon - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps

    def annealControllerEpsilon(self, stepCount, goal):
        self.controllerEpsilon[goal] = defaultEndEpsilon + (defaultControllerEpsilon[goal] - defaultEndEpsilon) * \
            (defaultAnnealSteps - max(0, stepCount - defaultRandomPlaySteps)) / defaultAnnealSteps
