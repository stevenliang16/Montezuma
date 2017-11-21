import argparse
import sys
import numpy as np

from environment import ALEEnvironment
from agent import Agent
from hdqn import Hdqn
from PIL import Image

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    goalExplain = ['top left door', 'top right door', 'middle ladder', 'lower left ladder', 'lower right ladder', 'key']
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
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
    args = parser.parse_args()
    env = ALEEnvironment(args.game, args)
    hdqn = Hdqn()
    print('loading weights')
    hdqn.loadWeight()
    print('weight loaded')
    agent = Agent(hdqn, range(8), range(6))
    # Probability of making random action is 0.1
    agent.setControllerEpsilon([0.1]*6)
    agent.setMetaEpsilon(0.1)
    while True:
        env.restart()
        for i in range(10):
            env.act(0)
        goalNum = 0
        while not env.isGameOver():
            goal = agent.selectTrueGoal(goalNum)
            print('predicted subgoal is: ' + str(goal) + ' ' + goalExplain[goal])
            while not env.isTerminal() and not env.goalReached(goal):
                state = env.getState()
                action = agent.selectMove(state, goal)
                #print ('selected action is: ' + str(actionMap[action]) + ' ' + actionExplain[actionMap[action]])
                #print('selected action is :' + str(actionExplain[action]))
                externalRewards = env.act(actionMap[action])
            if env.isTerminal() is False:
                goalNum = goalNum + 1
            else:
                # Re-initialize game if not game over
                if not env.isGameOver():
                    goalNum = 0
                    env.resetLife()
                    for i in range(10):
                        env.act(0)

if __name__ == "__main__":
    main()

