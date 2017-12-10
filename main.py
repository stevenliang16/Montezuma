import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
from environment import ALEEnvironment
from agent import Agent
from hdqn import Hdqn
from PIL import Image

# Constant defined here
maxStepsPerEpisode = 5000

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    # Initilization for tensor board
    session = tf.Session()
    tensorVar = tf.Variable(0)
    tensorVarLoss = tf.Variable(0, dtype = "float32")
    tensorVarMiddle = tf.Variable(0, dtype = "float32")
    tensorVarLowerRight = tf.Variable(0, dtype = "float32")
    tensorVarLowerLeft = tf.Variable(0, dtype = "float32")
    tensorVarKey = tf.Variable(0, dtype = "float32")
    
    tf.summary.scalar("reward", tensorVar)
    tf.summary.scalar("loss", tensorVarLoss)
    tf.summary.scalar("middle ladder", tensorVarMiddle)
    tf.summary.scalar("lower right ladder", tensorVarLowerRight)
    tf.summary.scalar("lower left ladder", tensorVarLowerLeft)
    tf.summary.scalar("key", tensorVarKey)
    sumWriterIntrinsic = tf.summary.FileWriter('./reward/intrinsic')
    sumWriterLoss = tf.summary.FileWriter('./reward/loss')
    sumWriterExternal = tf.summary.FileWriter('./reward/external')
    sumWriterMiddle = tf.summary.FileWriter('./reward/middleLadder')
    sumWriterLowerRight = tf.summary.FileWriter('./reward/lowerRightLadder')
    sumWriterLowerLeft = tf.summary.FileWriter('./reward/lowerLeftLadder')
    sumWriterKey = tf.summary.FileWriter('./reward/key')
    merged = tf.summary.merge_all()
    session.run(tf.initialize_all_variables())

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
    goalExplain = ['lower right ladder', 'lower left ladder', 'key']
    stepCount = 0
    goalSuccessTrack = [deque(), deque(), deque(), deque()] # deque in python is linkedlist, list is actually an array
    goalSuccessCount = [0, 0, 0, 0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=False)
    parser.add_argument("--frame_skip", default=4)
    parser.add_argument("--color_averaging", default=False)
    parser.add_argument("--random_seed")
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    parser.add_argument("--load_weight", default=False)
    parser.add_argument("--use_sparse_reward", type=str2bool, default=False)
    args = parser.parse_args()
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    annealComplete = False
    saveExternalRewardScreen = True
    env = ALEEnvironment(args.game, args)
    hdqn = Hdqn()
    
    # Initilize network and agent
    if (args.load_weight):
        defaultRandomPlaySteps = 200000
        print('loading weight')
        hdqn.loadWeight()
        print('loading weight complete')
        agent = Agent(hdqn, range(8), range(3))
    else:
        defaultRandomPlaySteps = 200000
        agent = Agent(hdqn, range(8), range(3))
    intrinsicRewardMonitor = 0
    externalRewardMonitor = 0
    for episode in range(80000):
        print("\n\n### EPISODE "  + str(episode) + "###")
        print("\n\n### STEPS "  + str(stepCount) + "###")
        # Restart the game
        env.restart()
        episodeSteps = 0
        # set goalNum to hardcoded subgoal
        lastGoal = -1
        while not env.isGameOver() and episodeSteps <= maxStepsPerEpisode:
            totalExternalRewards = 0 # NOT SURE IF IT SHOULD BE CLEARED HERE!
            stateLastGoal = env.getStackedState()
            # nextState = stateLastGoal
            goal = agent.selectGoal(stateLastGoal)
            if (len(goalSuccessTrack[goal]) > 100):
                firstElement = goalSuccessTrack[goal].popleft()
                goalSuccessCount[goal] -= firstElement
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = agent.selectMove(state, goal)
                externalRewards = env.act(actionMap[action])
                if (externalRewards != 0):
                    externalRewards = 1.0
                # Debugging
                if (saveExternalRewardScreen and externalRewards == 100):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save('keyGet.jpeg')
                    saveExternalRewardScreen = False
                stepCount += 1
                episodeSteps += 1
                # save the model every 50000 steps
                if (stepCount % 50000 == 0):
                    hdqn.saveWeight(stepCount)
                nextState = env.getStackedState()
                distanceReward = env.distanceReward(lastGoal, goal)
                # only assign intrinsic reward if the goal is reached and it has not been reached previously
                intrinsicRewards = agent.criticize(env.goalNotReachedBefore(goal) & env.goalReached(goal), actionMap[action], env.isTerminal(), distanceReward, args.use_sparse_reward)
                # Store transition and update network params
                exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.isTerminal())
                agent.store(exp, meta=False)
                
                # Do not update the network during random play
                if (stepCount >= defaultRandomPlaySteps):
                    if (stepCount == defaultRandomPlaySteps):
                        print('start training (random walk ends)')
                    if (stepCount % 4 == 0):
                        loss = agent.update(stepCount, meta=False)
                        agent.update(stepCount, meta=True)
                
                # Update external reward for D2
                totalExternalRewards += externalRewards + intrinsicRewards
                
                # Update data for visualization
                externalRewardMonitor += externalRewards
                intrinsicRewardMonitor += intrinsicRewards

            # Store meta controller's experience
            exp = MetaExperience(stateLastGoal, goal, totalExternalRewards, nextState, env.isTerminal())
            agent.store(exp, meta=True)
            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                goalSuccessTrack[goal].append(0)
                break
            elif env.goalReached(goal):
                goalSuccessTrack[goal].append(1)
                goalSuccessCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                # Training Visualization
                intrinsicPlot = session.run(merged, feed_dict={tensorVar: intrinsicRewardMonitor})
                sumWriterIntrinsic.add_summary(intrinsicPlot, stepCount)
                sumWriterIntrinsic.flush()
                externalPlot = session.run(merged, feed_dict={tensorVar: externalRewardMonitor})
                sumWriterExternal.add_summary(externalPlot, stepCount)
                sumWriterExternal.flush()
                lowerRightPlot = session.run(merged, feed_dict={tensorVarLowerRight: float(goalSuccessCount[0])/(0.1+len(goalSuccessTrack[0]))})
                sumWriterLowerRight.add_summary(lowerRightPlot, stepCount)
                sumWriterLowerRight.flush()
                lowerLeftPlot = session.run(merged, feed_dict={tensorVarLowerLeft: float(goalSuccessCount[1])/(0.1+len(goalSuccessTrack[1]))})
                sumWriterLowerLeft.add_summary(lowerLeftPlot, stepCount)
                sumWriterLowerLeft.flush()
                keyPlot = session.run(merged, feed_dict={tensorVarKey: float(goalSuccessCount[2])/(0.1+len(goalSuccessTrack[2]))})
                sumWriterKey.add_summary(keyPlot, stepCount)
                sumWriterKey.flush()
                lastGoal = goal
                # get key
                if goal == 2:
                    break
            else:
                goalSuccessTrack[goal].append(0)
                if not env.isGameOver():
                    lastGoal = -1
                    env.beginNextLife()

        if (not annealComplete):
            # Annealing 
            agent.annealMetaEpsilon(stepCount)
            agent.annealControllerEpsilon(stepCount, goal)

if __name__ == "__main__":
    main()
