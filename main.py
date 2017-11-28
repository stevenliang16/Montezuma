import argparse
import sys
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple
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
    tensorVarMiddle = tf.Variable(0)
    tensorVarLowerRight = tf.Variable(0)
    tensorVarLowerLeft = tf.Variable(0)
    tensorVarKey = tf.Variable(0)
    tf.summary.scalar("reward", tensorVar)
    tf.summary.scalar("middle ladder", tensorVarMiddle)
    tf.summary.scalar("lower right ladder", tensorVarLowerRight)
    tf.summary.scalar("lower left ladder", tensorVarLowerLeft)
    tf.summary.scalar("key", tensorVarKey)
    sumWriterIntrinsic = tf.summary.FileWriter('./reward/intrinsic')
    sumWriterExternal = tf.summary.FileWriter('./reward/external')
    sumWriterMiddle = tf.summary.FileWriter('./middleLadder')
    sumWriterLowerRight = tf.summary.FileWriter('./lowerRightLadder')
    sumWriterLowerLeft = tf.summary.FileWriter('./lowerLeftLadder')
    sumWriterKey = tf.summary.FileWriter('./key')
    merged = tf.summary.merge_all()
    session.run(tf.initialize_all_variables())

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
    goalExplain = ['middle ladder', 'lower right ladder', 'lower left ladder', 'key']
    stepCount = 0
    goalReachCount = [0, 0, 0, 0]
    goalAttemptCount = [0.1, 0.1, 0.1, 0.1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", type=str2bool, default=False)
    parser.add_argument("--frame_skip", default=4)
    #parser.add_argument("--repeat_action_probability", default=0.25)
    parser.add_argument("--color_averaging", default=False)
    parser.add_argument("--random_seed")
    #parser.add_argument("--record_screen_path", default="./record")
    #parser.add_argument("--record_sound_filename")
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
        defaultRandomPlaySteps = 100000
        print('loading weight')
        hdqn.loadWeight()
        print('loading weight complete')
        agent = Agent(hdqn, range(8), range(4))
    else:
        defaultRandomPlaySteps = 100000
        agent = Agent(hdqn, range(8), range(4))

    intrinsicRewardMonitor = 0
    externalRewardMonitor = 0
    for episode in range(30000):
        print("\n\n### EPISODE "  + str(episode) + "###")
        print("\n\n### STEPS "  + str(stepCount) + "###")
        # Restart the game
        env.restart()
        episodeSteps = 0
        # set goalNum to hardcoded subgoal
        goalNum = 0
        lastGoal = -1
        while not env.isGameOver() and episodeSteps <= maxStepsPerEpisode:
            totalExternalRewards = 0 # NOT SURE IF IT SHOULD BE CLEARED HERE!
            stateLastGoal = env.getStackedState()
            # goal = agent.selectGoal(env.getState())
            goal = agent.selectTrueGoal(goalNum)
            goalAttemptCount[goal] += 1
            #print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = agent.selectMove(state, goal)
                externalRewards = env.act(actionMap[action])
                # Debugging
                if (saveExternalRewardScreen and externalRewards == 100):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save('keyGet.jpeg')
                    saveExternalRewardScreen = False
                #print('reward is :' + str(externalRewards))
                stepCount += 1
                episodeSteps += 1
                # save the model every 50000 steps
                if (stepCount % 50000 == 0):
                    hdqn.saveWeight(stepCount)
                nextState = env.getStackedState()
                distanceReward = env.distanceReward(lastGoal, goal)
                # only assign intrinsic reward if the goal is reached and it has not been reached previously
                intrinsicRewards = agent.criticize(env.goalReached(goal), actionMap[action], env.isTerminal(), distanceReward, args.use_sparse_reward)
                ''' Debugging
                if (intrinsicRewards == 1.0):
                    print('subgoal reached')
                    im = Image.fromarray(np.squeeze(nextState))
                    im.save('goalReaced.jpeg')
                    sys.exit()
                '''
                # Store transition and update network params
                exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.isTerminal())
                agent.store(exp, meta=False)
                
                # Do not update the network during random play
                if (stepCount >= defaultRandomPlaySteps):
                    if (stepCount == defaultRandomPlaySteps):
                        print('start training (random walk ends)')
                    if (stepCount % 4 == 0):
                        agent.update(stepCount, meta=False)
                        # agent.update(stepCount, meta=True)
                
                # Update external reward for D2
                totalExternalRewards += externalRewards
                
                # Update data for visualization
                externalRewardMonitor += externalRewards
                intrinsicRewardMonitor += intrinsicRewards

            # Store meta controller's experience
            exp = MetaExperience(stateLastGoal, goal, totalExternalRewards, nextState, env.isTerminal())
            agent.store(exp, meta=True)
            
            # Update goal
            if episodeSteps > maxStepsPerEpisode:
                break
            elif env.goalReached(goal):
                lastGoal = goal
                goalReachCount[goal] += 1
                print('goal reached: ' + goalExplain[goal])
                '''
                if (goal == 2):
                    im = Image.fromarray(np.squeeze(env.getState()))
                    im.save(str(goalReachCount[goal])+'reachedLeftLadder.jpeg')
                '''
                goalNum = goalNum + 1
                if goalNum >= 4:
                   break
                # Training Visualization
                intrinsicPlot = session.run(merged, feed_dict={tensorVar: intrinsicRewardMonitor})
                sumWriterIntrinsic.add_summary(intrinsicPlot, stepCount)
                sumWriterIntrinsic.flush()
                externalPlot = session.run(merged, feed_dict={tensorVar: externalRewardMonitor})
                sumWriterExternal.add_summary(externalPlot, stepCount)
                sumWriterExternal.flush()
                middlePlot = session.run(merged, feed_dict={tensorVarMiddle: float(goalReachCount[0])/goalAttemptCount[0]})
                sumWriterMiddle.add_summary(middlePlot, stepCount)
                sumWriterMiddle.flush()
                lowerRightPlot = session.run(merged, feed_dict={tensorVarLowerRight: float(goalReachCount[1])/goalAttemptCount[1]})
                sumWriterLowerRight.add_summary(lowerRightPlot, stepCount)
                sumWriterLowerRight.flush()
                lowerLeftPlot = session.run(merged, feed_dict={tensorVarLowerLeft: float(goalReachCount[2])/goalAttemptCount[2]})
                sumWriterLowerLeft.add_summary(lowerLeftPlot, stepCount)
                sumWriterLowerLeft.flush()
                keyPlot = session.run(merged, feed_dict={tensorVarKey: float(goalReachCount[3])/goalAttemptCount[3]})
                sumWriterKey.add_summary(keyPlot, stepCount)
                sumWriterKey.flush()
            else:
                if not env.isGameOver():
                    goalNum = 0
                    lastGoal = -1
                    env.beginNextLife()
                               
            #intrinsicRewardMonitor = 0
            #externalRewardMonitor = 0

        if (not annealComplete):
            #Annealing 
            # agent.annealMetaEpsilon(stepCount)
            agent.annealControllerEpsilon(stepCount, goal)

if __name__ == "__main__":
    main()
