from __future__ import print_function

import random
import gym
import numpy as np
import tensorflow as tf
from keras import backend as K
import models

sess = tf.Session()
K.set_session(sess)

run_name = "mentor_qnet"
summary_name = run_name + "_accuracy"
model_save_name = run_name + ".h5"

# some params
max_episodes = 2500
batch_size = 1
epsilon = 0.1
episode_len = 1000
discount = True
standardize = True

env = gym.make('CartPole-v0')
episode_rewards = []

# helper Methods
def predict(model, input):
    # print (model)
    # print (input)
    # print (model.inbound_nodes)
    # print (len(model.layers))
    # for l in model.layers:
    #     print (l, l.inbound_nodes)
    input = input[np.newaxis,:]
    return model.predict(input)


def sample(x):
    assert isinstance(x, np.ndarray)
    x = np.squeeze(x)
    assert x.ndim == 1
    # renormalize to avoid 'does not sum to 1 errors'
    return np.random.choice(len(x), 1, p=x/x.sum())


def rollout_env_with_policy(env, policy, episode_len=np.inf, render=True, verbose=False):
    """
    Runs environment to completion and returns reward under given policy
    Returns the sequence of rewards, states, raw actions (direct from the policy),
        and processed actions (actions sent to the env)
    """
    ep_rewards = []
    ep_states = []
    ep_raw_actions = []
    ep_processed_actions = []

    # episode_reward = 0.0
    done = False
    obs = env.reset()
    episode_itr = 0
    while not done and episode_itr < episode_len:
        if render:
            env.render()

        ep_states.append(obs)
        # print (policy)
        # print (obs)
        action = sess.run(policy, feed_dict={states: obs.reshape(1, len(obs))})
        # action = predict(policy, obs) #policy.predict(obs)

        # import sys
        # sys.exit()

        ep_raw_actions.append(action)
        action = sample(action)
        action = int(action)
        ep_processed_actions.append(action)

        obs, reward, done, _ = env.step(action)
        ep_rewards.append(reward)

        episode_itr += 1

    if verbose:
        print ('Game finished, reward: %f' % (sum(ep_rewards)))

    return ep_states, ep_raw_actions, ep_processed_actions, ep_rewards


def discount_rewards(rewards, gamma=0.9):
    """
    Take 1D float array of rewards and compute the sum of discounted rewards
    at each timestep
    """
    discounted_r = np.zeros_like(rewards)
    for i in xrange(rewards.size):
        rew_sum = 0.0
        for j in xrange(i, rewards.size):
            rew_sum += rewards[j] * gamma ** j
        discounted_r[i] = rew_sum
    return discounted_r


def standardize_rewards(rewards):
    """
    Subtract the mean and divide by the stddev of the given rewards
    """
    rew_mean = np.mean(rewards)
    rew_std = np.std(rewards)
    rewards -= rew_mean
    if rew_std != 0:
        rewards /= rew_std
    return rewards


# Policy gradient operations

obs_shape = tuple([None]+list(env.observation_space.shape))
states = tf.placeholder(tf.float32, shape=obs_shape) #(None, env.observation_space.shape[0]))
# states = models.obs_cartpole
actions = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
rewards = tf.placeholder(tf.float32, shape=(None))

# n_input = 4
# n_hidden_1 = 8
# n_hidden_2 = 8
# n_classes = 2
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
# # Hidden layer with RELU activation
# layer_1 = tf.add(tf.matmul(states, weights['h1']), biases['b1'])
# layer_1 = tf.nn.relu(layer_1)
# # Hidden layer with RELU activation
# layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
# layer_2 = tf.nn.relu(layer_2)
# # Output layer with softmax activation
# out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
# probs = tf.nn.softmax(out_layer)


approx = models.build_mentor_model_dqn(states)
probs = approx.output #approx(states) #.output #approx_pred
action_probs = tf.mul(probs, actions)
reduced_action_probs = tf.reduce_sum(action_probs, reduction_indices=[1])
logprobs = tf.log(reduced_action_probs)

# vanilla gradient = mul(sum(logprobs * rewards))
L = -tf.reduce_sum(tf.mul(logprobs, rewards))
opt = tf.train.AdamOptimizer(0.01)

# do gradient update separately so do apply custom function to gradients?
grads_and_vars = opt.compute_gradients(L)
clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in grads_and_vars]
update = opt.apply_gradients(clipped_grads_and_vars)

sess.run(tf.initialize_all_variables())

episode = 0
while episode < max_episodes:
    ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rollout_env_with_policy(env,
                                                                                          probs,
                                                                                          episode_len=episode_len)

    total_reward = sum(ep_rewards)
    episode_rewards.append(total_reward)

    if discount:
        ep_rewards = discount_rewards(np.array(ep_rewards))

    formatted_actions = np.zeros((len(ep_raw_actions), env.action_space.n))
    for i in range(len(ep_processed_actions)):
        formatted_actions[i][ep_processed_actions[i]] = 1.0

    formatted_rewards = ep_rewards
    if standardize:
        formatted_rewards = standardize_rewards(formatted_rewards)

    sess.run(update, feed_dict={actions: formatted_actions,
                                states: ep_states,
                                rewards: formatted_rewards})

    print ("Episode: ", episode, ", reward: ", total_reward)
    if episode % 100 == 0:
        print ("Saving model")
        approx.save("mentor_polgrad.h5")

    episode += 1



import matplotlib.pyplot as plt
rewards = np.array(episode_rewards)
plt.plot(rewards)
plt.show()
