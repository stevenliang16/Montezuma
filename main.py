import argparse
from collections import namedtuple
from environment import ALEEnvironment
from agent import Agent
from hdqn import Hdqn

# Constant defined here
anneal_factor = (1.0-0.1)/12000

def main():
    print("Annealing factor: " + str(anneal_factor))
    stepCount = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="montezuma_revenge.bin")
    parser.add_argument("--display_screen", action="store_true", default=True)
    parser.add_argument("--frame_skip", default=1)
    parser.add_argument("--repeat_action_probability", default=0.25)
    parser.add_argument("--color_averaging", default=False)
    parser.add_argument("--random_seed")
    parser.add_argument("--record_screen_path", default="./record")
    parser.add_argument("--record_sound_filename")
    parser.add_argument("--minimal_action_set", default=False)
    parser.add_argument("--screen_width", default=84)
    parser.add_argument("--screen_height", default=84)
    args = parser.parse_args()
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state", "done"])
    env = ALEEnvironment(args.game, args)
    hdqn = Hdqn()
    agent = Agent(hdqn, range(18), range(6))
    for episode_thousand in range(12):
        for episode in range(1000):
            print("\n\n### EPISODE "  + str(episode_thousand*1000 + episode) + "###")
            env.reset()
            goal = agent.selectGoal(env.getScreen())
            while not env.isTerminal():
                totalExternalRewards = 0
                stateLastGoal = env.getScreen()
                while not env.isTerminal() and not env.goalReached(goal):
                    state = env.getScreen()
                    action = agent.selectMove(state, goal)
                    externalRewards = env.act(action)
                    stepCount += 1
                    nextState = env.getScreen()
                    intrinsicRewards = agent.criticize(env.goalReached(goal))

                    # Store transition and update network params
                    exp = ActorExperience(state, goal, action, intrinsicRewards, nextState, env.isTerminal())
                    agent.store(exp, meta=False)
                    # Do not update the network during random play
                    if (stepCount >= 10000):
                        agent.update(meta=False)
                        agent.update(meta=True)
                    totalExternalRewards += externalRewards

                # Store meta controller's experience
                exp = MetaExperience(stateLastGoal, goal, totalExternalRewards, nextState, env.isTerminal())
                agent.store(exp, meta=True)
                
                # Upodate goal
                if env.isTerminal() is False:
                    goal = agent.selectGoal(env.getScreen(), agent.metaEpsilon)

                #Annealing 
                agent.annealMetaEpsilon(stepCount)
                agent.annealControllerEpsilon(stepCount, goal)
                


if __name__ == "__main__":
    main()
