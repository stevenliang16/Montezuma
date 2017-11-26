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
defaultRandomPlaySteps = 100000
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
    parser.add_argument("--frame_skip", default=80)
    #parser.add_argument("--repeat_action_probability", default=0.25)
    parser.add_argument("--color_averaging", default=False)
    parser.add_argument("--random_seed")
    #parser.add_argument("--record_screen_path", default="./record")
    #parser.add_argument("--record_sound_filename")
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    args = parser.parse_args()
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    env = ALEEnvironment(args.game, args)
    hdqn = Hdqn()
    agent = Agent(hdqn, range(8), range(6))
    # set goalNum to hardcoded subgoal
    goalNum = 0
    intrinsicRewardMonitor = 0
    externalRewardMonitor = 0
    env.act(12)
    # for i in range(100):
    #     env.act(0)
    #     print(env.isTerminal())
    print(env.isTerminal())

if __name__ == "__main__":
    main()
