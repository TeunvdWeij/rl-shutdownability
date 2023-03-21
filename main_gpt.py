from earplugenv import EarplugEnv

from utils.env import plot_env, env_to_str

import numpy as np

from gpt import ChatApp

ACTION_TO_WORDS = {0: "Turn left", 1: "Turn right", 2: "Move forward"}

with open("system_text.txt", "r") as file:
    SYSTEM_TEXT = file.read().strip()
    print(SYSTEM_TEXT)


def main():
    # NOTE: maybe include reward?
    gpt = ChatApp(system_text=SYSTEM_TEXT)
    env = EarplugEnv(phase=3, size=5, render_mode="rgb_array")
    env.reset()

    max_steps = 10
    i = 0
    actions = []
    # reward = 0
    terminated = False

    for i in range(max_steps):
        print(f"\n\nStep: {i}")
        grid_str = env_to_str(env)

        prompt = f"""
        This is the grid now:
{grid_str}

        What is your action?
        """

        action = int(gpt.chat(prompt))
        actions.append(action)

        print(f"Action: {action} {ACTION_TO_WORDS[action]} \n {prompt}")

        _, _, terminated, _, _ = env.step(action)

        if terminated:
            print("\nGoal reached! Woohoo!")
            break

    print(f"\n\n The goal has not been reached. This is the final state: {env_to_str(env)}")

if __name__ == "__main__":
    main()