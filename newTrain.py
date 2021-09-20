import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks import q_network

from env import EnduranceEnv

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")

# hyper params
num_iterations = 1800  # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 50  # @param {type:"integer"}

num_eval_episodes = 150  # @param {type:"integer"}
eval_interval = 600  # @param {type:"integer"}

# environment
eval_py_env = EnduranceEnv(40000, "eval")
train_py_env = EnduranceEnv(40001, "train")
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

# q network
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(eval_env.action_spec())
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

# agnet
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


def compute_avg_return(environment, policy, num_episodes=10):
    print("[compute_avg_return] start, episode num = {0}".format(num_episodes))
    total_return = 0.0
    for e in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        i = 0
        while not time_step.is_last():
            #print("[compute_avg_return] episode: {0} step: {1}".format(e, i))
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            i = i + 1
        total_return += episode_return

    avg_return = total_return / num_episodes
    rtn = avg_return.numpy()[0]
    print("[compute_avg_return] complete, avg_rtn={0}".format(rtn))
    return rtn


# replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    print("[collect_data] start, step_num={0}".format(steps))
    for i in range(steps):
        collect_step(env, policy, buffer)
    print("[collect_data] complete")


print("--- fill replay buffer start")
if os.path.exists("checkpoint/checkpoint"):
    print("checkpoint exist! open existing replay buffer")
    cp = tf.train.Checkpoint(rb=replay_buffer)
    cp.restore("checkpoint/replay_buffer-1")
    replay_buffer.get_next()
else:
    collect_data(train_env, random_policy,
                 replay_buffer, initial_collect_steps)
    cp = tf.train.Checkpoint(rb=replay_buffer)
    cp.save("checkpoint/replay_buffer")
print("--- fill replay buffer complete")

# dataset
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for i in range(num_iterations):
    #print("[training iteration]: {0}".format(i))
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(train_env, agent.collect_policy,
                 replay_buffer, collect_steps_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.show()
plt.savefig("graph.jpg")
