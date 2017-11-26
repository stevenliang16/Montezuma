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
    tf.summary.scalar("reward", tensorVar)
    sumWriterIntrinsic = tf.summary.FileWriter('./reward/intrinsic')
    sumWriterExternal = tf.summary.FileWriter('./reward/external')
    merged = tf.summary.merge_all()
    session.run(tf.initialize_all_variables())

    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
    goalExplain = ['top left door', 'top right door', 'middle ladder', 'lower left ladder', 'lower right ladder', 'key']
    stepCount = 0
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
    env = ALEEnvironment(args.game, args)
    hdqn = Hdqn()
    
    # Initilize network and agent
    if (args.load_weight):
        defaultRandomPlaySteps = 100000
        print('loading weight')
        hdqn.loadWeight()
        print('loading weight complete')
        agent = Agent(hdqn, range(8), range(6))
    else:
        defaultRandomPlaySteps = 100000
        agent = Agent(hdqn, range(8), range(6))

    intrinsicRewardMonitor = 0
    externalRewardMonitor = 0
    for episode in range(30000):
        # save the model every 100 episode
        if (episode % 500 == 0):
            hdqn.saveWeight(episode)
        print("\n\n### EPISODE "  + str(episode) + "###")
        print("\n\n### STEPS "  + str(stepCount) + "###")
        # Restart the game
        env.restart()
        episodeSteps = 0
        # set goalNum to hardcoded subgoal
        goalNum = 0
        # initial last goal
        lastGoal = -1
        while not env.isGameOver() and episodeSteps <= maxStepsPerEpisode:
            totalExternalRewards = 0 # NOT SURE IF IT SHOULD BE CLEARED HERE!
            stateLastGoal = env.getStackedState()
            # goal = agent.selectGoal(env.getState())
            # goal = agent.selectTrueGoal(goalNum)
            goal = 2
            print('predicted subgoal is: ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal) and episodeSteps <= maxStepsPerEpisode:
                state = env.getStackedState()
                action = agent.selectMove(state, goal)

                #print('selected action is :' + str(actionExplain[action]))
                externalRewards = env.act(actionMap[action])
                #print('reward is :' + str(externalRewards))
                if(externalRewards != 0):
                    exState = env.getState()
                    im = Image.fromarray(np.squeeze(exState))
                    im.save('getExternalRewards.jpeg')
                stepCount += 1
                episodeSteps += 1
                nextState = env.getStackedState()
                distanceReward = env.distanceReward(lastGoal, goal)
                # only assign intrinsic reward if the goal is reached and it has not been reached previously
                intrinsicRewards = agent.criticize(env.goalReached(goal), actionMap[action], env.isLifeLost(), distanceReward, args.use_sparse_reward)
                ''' Debugging
                if (intrinsicRewards == 1.0):
                    print('subgoal reached')
                    im = Image.fromarray(np.squeeze(nextState))
                    im.save('goalReaced.jpeg')
                    sys.exit()
                '''
                '''
                if (intrinsicRewards == 1.0):
                    print(stepCount)
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
            elif env.isTerminal() is False:
                # lastGoal = goal
                # goalNum = goalNum + 1
                # if goalNum >= 4:
                #     break
                print "reached middle ladder bottom"
                break
            else:
                # Re-initialize game if not game over
                if not env.isGameOver():
                    goalNum = 0
                    lastGoal = -1
                    env.beginNextLife()
                               
            # Training Visualization
            intrinsicPlot = session.run(merged, feed_dict={tensorVar: intrinsicRewardMonitor})
            sumWriterIntrinsic.add_summary(intrinsicPlot, stepCount)
            sumWriterIntrinsic.flush()
            externalPlot = session.run(merged, feed_dict={tensorVar: externalRewardMonitor})
            sumWriterExternal.add_summary(externalPlot, stepCount)
            sumWriterExternal.flush()
            #intrinsicRewardMonitor = 0
            #externalRewardMonitor = 0

        if (not annealComplete):
            #Annealing 
            agent.annealMetaEpsilon(stepCount)
            agent.annealControllerEpsilon(stepCount, goal)

if __name__ == "__main__":
    main()
