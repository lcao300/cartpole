import time
import numpy as np
import gym
from tqdm import tqdm


def make_env():
    """
    Makes and sets up CartPole environment

        Parameters:
            none

        Returns:
            env (obj): environment object
    """
    np.random.seed(42)
    env = gym.make('CartPole-v0').unwrapped
    env.seed(42)
    env.reset()
    return env


def get_discrete_state(state):
    """
    Convert the continuous state to discrete state in CartPole

        Parameters:
            state (tuple): continuous state

        Returns:
            discrete_state (tuple): discrete state
    """
    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(np.int))


def q_learning_cart(env):  # add your own arguments based on your implementation
    """
    Performs Q-learning for the CartPole environment

        Parameters:
            env (obj): environment object

        Returns:
            Q (arr): array of q values
    """
    # number of actions
    num_actions = env.action_space.n

    # create the discrete state space
    state_space = [30, 30, 50, 50]

    # initialize Q-table with random numbers
    q_table = np.random.uniform(low=0, high=1, size=(state_space + [num_actions]))

    # set parameters
    alpha = 0.05
    gamma = 0.95
    epsilon_0 = 1.0
    epsilon_min = 0.005
    lambda_param = 0.000001
    num_episodes = 40000
    num_steps = 700

    # loop through for number of episodes
    for episode in tqdm(range(num_episodes)):
        # init s
        discrete_state = get_discrete_state(env.reset())

        # calculate epsilon
        epsilon = max(epsilon_0 * np.exp(-1 * lambda_param * episode), epsilon_min)
        done = False

        # loop until terminal state
        for step in range(num_steps):
            # e-greedy
            prob = np.random.rand()
            if (prob < epsilon) or (np.sum(q_table[discrete_state]) == 0):
                action = np.random.choice(num_actions)

            else:
                action = np.argmax(q_table[discrete_state])

            # take action and observe
            observation,reward,done,_ = env.step(action)
            discrete_ob = get_discrete_state(observation)

            # update Q
            q_table[discrete_state][action] = q_table[discrete_state][action] + alpha * \
                (reward + gamma*max(q_table[discrete_ob].tolist()) - \
                    q_table[discrete_state][action])

            # move state
            discrete_state = discrete_ob
            if done:
                break

    return q_table


def test():
    """
    Runs Q-learning algorithm from q_table.py

        Parameters:
            none

        Returns:
            none
    """
    env = make_env()

    # load the q-table
    print('Load the q-table...')
    q_table = np.load('q_table.npy')

    print('Test your policy from q-learning for CartPole...')
    num_test = 10
    t_total = 0
    step_total = 0
    if np.shape(np.array(q_table)) != (30, 30, 50, 50, env.action_space.n):
        print('Q-table is in the wrong shape.')
    else:
        for i in range(num_test):
            env.reset()
            current_state = get_discrete_state(env.reset())
            time_0 = time.time()
            step = 0
            while True:
                step += 1
                env.render()
                action = np.argmax(q_table[current_state])
                next_state, _, done, _ = env.step(action)
                current_state = get_discrete_state(next_state)

                if done or step >= 2000:
                    time_1 = time.time()
                    t_total += time_1 - time_0
                    step_total += step
                    print('time: %.2f' %(time_1 - time_0), end=' ')
                    print('step:', step)
                    break
        env.close()

    print('Average time:', t_total / num_test)
    print('Average step:', step_total / num_test)


def train():
    """
    Wrapper for q_learning_cart

        Parameters:
            none

        Returns:
            none
    """
    env = make_env()
    print('Run q-learning...')
    q_table = q_learning_cart(env)

    # save the q-table
    print('Save the q-table...')
    np.save('q_table.npy', q_table)

    print('Test your policy from q-learning for CartPole...')

    test()
