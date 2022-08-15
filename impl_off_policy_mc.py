import random

import matplotlib.pyplot as plt

from impl_env_racetrack import RacetrackEnv


def generate_episode(env, policy):
    print()
    episode_list = []
    state = env.reset()
    done = False
    iters = 0
    while not done:
        iters += 1
        action = (random.choice(env.get_actions()), random.choice(env.get_actions()))
        new_state, reward, done = env.step(state, action)
        episode_list.append((state, action, reward, done))
        state = new_state

        print(f'\rcreating episode [{iters}]', end='')
    return episode_list


def update_policy(env, policy, q_func, state):
    max_dict = {}
    for action_x in env.get_actions():
        for action_y in env.get_actions():
            state_action = (state[0], state[1], state[2], state[3], action_x, action_y)
            q_val = q_func[state_action]
            max_dict[(action_x, action_y)] = q_val
    best_action = max(max_dict, key=max_dict.get)
    # if best_action != (0, 0):
        # print(f'{best_action}, val: {max_dict[best_action]}')
    policy[state] = best_action
    return best_action


def examine_policy(env, policy):
    print()
    episode_list = []
    state = env.reset()
    for iters in range(1000):
        iters += 1
        # action = (random.choice(env.get_actions()), random.choice(env.get_actions()))
        action = policy[state]
        new_state, reward, done = env.step(state, action)
        episode_list.append((state, action, reward, done))
        state = new_state

        print(f'\rexamine policy [{iters}], action: {action}', end='')

        if done:
            break
    x_list = [step[0][0] for step in episode_list]
    y_list = [step[0][1] for step in episode_list]
    plt.plot(x_list, y_list)
    plt.xlim([0, env.side_size])
    plt.ylim([0, env.side_size])
    plt.show()


def off_policy_mc(env, q_func, c_func, policy, episodes=100):
    for i in range(episodes):
        # generate episode
        episode_list = generate_episode(env, policy)

        # update functions
        G = 0
        W = 1

        print()
        iters = 0
        for step in reversed(episode_list):
            state, action, reward, done = step
            G = GAMMA * G + reward
            state_action = (state[0], state[1], state[2], state[3], action[0], action[1])
            c_func[state_action] += W
            c_val = c_func[state_action]
            q_val = q_func[state_action]
            q_func[state_action] += (W/c_val) * (G - q_val)
            best_action = update_policy(env, policy, q_func, state)
            if action != best_action:
                break
            W = W * 9

            print(f'\rupdate functions [{iters}], episodes: [{i}]', end='')

        examine_policy(env, policy)

    return q_func, c_func, policy


def main():
    env = RacetrackEnv()
    # env.render()
    q_func = env.init_q_func()
    c_func = env.init_c_func()
    policy = env.init_policy()
    q_func, c_func, policy = off_policy_mc(env, q_func, c_func, policy)


if __name__ == '__main__':
    GAMMA = 0.9
    main()





