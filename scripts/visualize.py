import argparse
import numpy

import utils
from utils import device


def visualize(args):

    # Set seed for all randomness sources

    utils.seed(args.get('seed'))

    # Set device

    print(f"Device: {device}\n")

    # Load environment
    env = args.get('envs')

    # env = utils.make_env(args.get('env'), args.get('seed'), render_mode="human")
    # for _ in range(args.get('shift')):
    #     env.reset()
    # print("Environment loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.get('model'))
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        argmax=args.get('argmax'), use_memory=args.get('memory'), use_text=args.get('text'))
    print("Agent loaded\n")

    # Run the agent

    if args.get('gif'):
        from array2gif import write_gif

        frames = []

    # Create a window to view the environment
    env.render()

    for episode in range(int(args.get('episodes'))):

        obs, _ = env.reset()

        while True:
            env.render()
            if args.get('gif'):
                frames.append(numpy.moveaxis(env.get_frame(), 2, 0))

            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.analyze_feedback(reward, done)

            if done: # or env.window.closed: #NOTE: throws error
                break

        if env.window.closed:
            break

    if args.get('gif'):
        print("Saving gif... ", end="")
        write_gif(numpy.array(frames), args.get('gif')+".gif", fps=1/args.get('pause'))
        print("Done.")
