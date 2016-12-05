from __future__ import print_function

import random
import gym
import numpy as np
import tensorflow as tf
from keras import backend as K
import models
import probe_ops
import learning_rates
learning_rates.convergence = 2500
learning_rates.num_epochs_to_learn_representation = 200

sess = tf.Session()
K.set_session(sess)

run_name = "mentee_polgrad"
summary_name = run_name + "_accuracy"
model_save_name = run_name + ".h5"

# some params
max_episodes = 5000
batch_size = 1
epsilon = 0.1
episode_len = 1000
discount = True
standardize = True
mentee_mode = 'obedient'
temperature = 0.9
# list of probes between the mentor and mentee by layer; 0-indexed
# the output probe does not need to be specified
probes = [
    (0, 0),
]

env = gym.make('CartPole-v0')
episode_rewards = []

# helper Methods

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
        # action = predict(policy, obs) #policy.predict(obs)
        action = sess.run(policy, feed_dict={states: obs.reshape(1, len(obs))})

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
actions = tf.placeholder(tf.float32, shape=(None, env.action_space.n))
rewards = tf.placeholder(tf.float32, shape=(None))

mentee_approx = models.build_mentee_model_dqn(states)
mentee_probs = mentee_approx.output
action_probs = tf.mul(mentee_probs, actions)
reduced_action_probs = tf.reduce_sum(action_probs, reduction_indices=[1])
logprobs = tf.log(reduced_action_probs)

# vanilla gradient = mul(sum(logprobs * rewards))
L = -tf.reduce_sum(tf.mul(logprobs, rewards))
learning_rate = tf.placeholder(tf.float32, shape=[])
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

# do gradient update separately so do apply custom function to gradients?
grads_and_vars = opt.compute_gradients(L)
clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in grads_and_vars]
update = opt.apply_gradients(clipped_grads_and_vars)

print (len(clipped_grads_and_vars))
mentor_approx = models.build_mentor_model_dqn(states, load=True)
mentor_probs = mentor_approx.output

probe_gradients = probe_ops.get_gradient_ops(probes, mentee_approx, mentor_approx, states, batch_size, temperature)

sess.run(tf.initialize_all_variables())

episode = 0
while episode < max_episodes:
    ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rollout_env_with_policy(env,
                                                                                          mentee_probs,
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

    # gradients computed according to policy gradient theorem
    gradients = [sess.run(g, feed_dict={actions: formatted_actions,
                                        states: ep_states,
                                        rewards: formatted_rewards}) for g, v, in clipped_grads_and_vars]

    # compute all probes (w/o the softmax probe)
    computed_probe_gradients = []
    for j in range(len(probe_gradients)-1):
        probe_grad = []
        probe_grad_op = probe_gradients[j]
        for g in probe_grad_op:
            if g is not None:
                probe_grad.append(sess.run(g, feed_dict={states: ep_states}))
            else:
                probe_grad.append(None)
        computed_probe_gradients.append(probe_grad)

    # compute gradients for softmax probe
    computed_probe_out_gradients = [sess.run(g, feed_dict={states: ep_states}) for g in probe_gradients[-1]]

    n = learning_rates.compute_n(episode)
    # scale by 1.0/n because these param are a*lr and lr will be applied in the gradient update
    a = learning_rates.compute_eta_alpha(episode, mentee_mode)*1.0/n
    b = learning_rates.compute_eta_beta(episode, mentee_mode)*1.0/n
    g = learning_rates.compute_eta_gamma(episode, mentee_mode)*1.0/n

    for j in range(len(gradients)):
        # set gradients for variable j
        gradients[j] = a*gradients[j]

        # sum the probes
        for k in range(len(computed_probe_gradients)):
            probe_grad = computed_probe_gradients[k]
            if (probe_grad[j] is not None): # if there is a gradient from probe k for var j
                gradients[j] += b*probe_grad[j]

        # add the output softmax probe
        gradients[j] += g*computed_probe_out_gradients[j]

    # apply grads
    grads_n_vars = [(gradients[x], clipped_grads_and_vars[x][1]) for x in range(len(clipped_grads_and_vars))]
    # sess.run(opt.apply_gradients(grads_n_vars), feed_dict={learning_rate: n})
    sess.run(opt.apply_gradients(grads_n_vars), feed_dict={learning_rate: n})

    # could also run mentor for sample obs, and then have good actions
    # then do output probe?
    # 1) Get gradients of mentee according to pg theorem
    # 2) Get loss of output probes across all actions in the episode
    # 3) Get loss of all hidden layer probes across all actions in the episode

    print ("Episode: ", episode, ", reward: ", total_reward)
    if episode % 100 == 0:
        print ("Saving model")
        mentee_approx.save("mentee_polgrad.h5")

    episode += 1



import matplotlib.pyplot as plt
rewards = np.array(episode_rewards)
plt.plot(rewards)
plt.show()
