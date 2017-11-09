import argparse
import sys
from collections import namedtuple
from environment import ALEEnvironment
from agent import Agent
from hdqn import Hdqn

# Constant defined here
anneal_factor = (1.0-0.1)/12000

def main():
    actionMap = [0, 1, 2, 3, 4, 5, 11, 12]
    actionExplain = ['no action', 'jump', 'up', 'right', 'left', 'down', 'jump right', 'jump left']
    print("Annealing factor: " + str(anneal_factor))
    stepCount = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", action="store_true", default=False)
    parser.add_argument("--frame_skip", default=1)
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
    for episode_thousand in range(100):
        # save the model every 1000 episode
        hdqn.saveWeight(episode_thousand)
        for episode in range(1000):
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            env.reset()
            goal = agent.selectGoal(env.getState())
            print(goal)
            print(env.isTerminal())
            while not env.isTerminal():
                totalExternalRewards = 0
                stateLastGoal = env.getState()
                while not env.isTerminal() and not env.goalReached(goal):
                    state = env.getState()
                    action = agent.selectMove(state, goal)
                    print('selected action is :' + str(actionExplain[action]))
                    externalRewards = env.act(actionMap[action])
                    print('reward is :' + str(externalRewards))
                    stepCount += 1
                    nextState = env.getState()
                    intrinsicRewards = agent.criticize(env.goalReached(goal))
                    # Store transition and update network params
                    exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.isTerminal())
                    agent.store(exp, meta=False)
                    # Do not update the network during random play
                    if (stepCount >= 100000):
                        if (stepCount == 100000):
                            print('start training (random walk ends)')
                        agent.update(meta=False)
                        agent.update(meta=True)
                    totalExternalRewards += externalRewards

                # Store meta controller's experience
                exp = MetaExperience(stateLastGoal, goal, totalExternalRewards, nextState, env.isTerminal())
                agent.store(exp, meta=True)
                
                # Update goal
                if env.isTerminal() is False:
                    goal = agent.selectGoal(env.getState())

                #Annealing 
                agent.annealMetaEpsilon(stepCount)
                agent.annealControllerEpsilon(stepCount, goal)
                


if __name__ == "__main__":
    main()
