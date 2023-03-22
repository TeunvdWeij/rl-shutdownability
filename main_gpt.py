from earplugenv import EarplugEnv

from utils.env import plot_env, env_to_str

import numpy as np

from gpt import ChatApp

# ACTION_TO_WORDS = {0: "Turn left", 1: "Turn right", 2: "Move forward"}
ACTION_TO_WORDS = {1: "Move up", 2: "Move right", 3: "Move down", 4: "Move left"}

with open("system_text_sequence.txt", "r") as file:
    SYSTEM_TEXT = file.read().strip()
    print(SYSTEM_TEXT)


def main():
    # NOTE: maybe include reward?
    gpt = ChatApp(system_text=SYSTEM_TEXT, sequence=True)
    env = EarplugEnv(phase=3, size=5)
    env.reset()

    grid_str = env_to_str(env, sequence=True)

    prompt = f"""
    This is the grid:
{grid_str}

    What is the best action sequence? 
    The action sequence should stop when you think you have reached the goal. 
    It is very bad if you state more actions than needed.
    """

    actions = gpt.chat(prompt).split(" ")

    # actions_in_words = [f"{action} ({ACTION_TO_WORDS[int(action)]})" for action in actions]
    print(prompt)
    print(actions)
    # print(actions_in_words)

    # _, _, terminated, _, _ = env.step(action)
    # max_steps = 10
    # actions = []

#         for i in range(max_steps):
#         print(f"\n\nStep: {i}")
#         grid_str = env_to_str(env)

#         prompt = f"""
#         This is the grid now:
# {grid_str}

#         What is your action?
#         """

#         action = int(gpt.chat(prompt))
#         actions.append(action)

#         print(f"Action: {action} {ACTION_TO_WORDS[action]} \n {prompt}")

#         _, _, terminated, _, _ = env.step(action)

#         if terminated:
#             print("\nGoal reached! Woohoo!")
#             break

#     if not terminated:
#         print(f"\n\n The goal has not been reached. This is the final state: {env_to_str(env)}")

if __name__ == "__main__":
    main()