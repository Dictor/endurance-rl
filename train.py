import matplotlib.pyplot as plt
import os
from datetime import datetime

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks import q_network

from env import EnduranceEnv

from trainUtil import *

# prepare path
if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")
has_checkpoint = os.path.exists("checkpoint/checkpoint")

# hyper params
num_iterations = 10000  # @param {type:"integer"}

initial_collect_steps = 10000  # @param {type:"integer"}
collect_steps_per_iteration = 4  # @param {type:"integer"}
replay_buffer_max_length = 10000  # @param {type:"integer"}

batch_size = 16  # @param {type:"integer"}
learning_rate = 0.00001  # @param {type:"number"}
log_interval = 100  # @param {type:"integer"}

num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 500  # @param {type:"integer"}

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

# agent
global_step = tf.compat.v1.train.get_or_create_global_step()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)
agent.initialize()
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


if not has_checkpoint:
    print("no existing checkpoint! create replay buffer.")
    collect_data(train_env, random_policy,
                 replay_buffer, initial_collect_steps)

# train checkpointer
train_checkpointer = common.Checkpointer(
    ckpt_dir="checkpoint",
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()

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
    try:
        # print("[training iteration]: {0}".format(i))
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
    except KeyboardInterrupt:
        print("key int! exit loop")
        break

# save agent
train_checkpointer.save(global_step)

iterations = range(0, eval_interval*len(returns), eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig("graph{0}.jpg".format(datetime.now().strftime("%y%d%m-%H_%M_%S")))
plt.show()
