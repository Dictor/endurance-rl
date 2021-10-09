
import os

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks import q_network

from env import EnduranceEnv

from trainUtil import *

# hyper parameter
learning_rate = 0.00001  # @param {type:"number"}

# prepare path
if (not os.path.exists("checkpoint")):
    print("cannot found checkpoint directory!")
    exit

# environment
eval_py_env = EnduranceEnv(40000, "eval")
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# q network
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(eval_env.action_spec())
q_net = q_network.QNetwork(
    eval_env.observation_spec(),
    eval_env.action_spec(),
    fc_layer_params=fc_layer_params
)

# agent
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    eval_env.time_step_spec(),
    eval_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()
random_policy = random_tf_policy.RandomTFPolicy(eval_env.time_step_spec(),
                                                eval_env.action_spec())

# global step
global_step = tf.compat.v1.train.get_or_create_global_step()

# checkpointer
train_checkpointer = common.Checkpointer(
    ckpt_dir="checkpoint",
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    global_step=global_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()
print("global step : {0}".format(global_step.numpy()))
print("policy: {0}".format(agent.policy))
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, 3)
print("{0} episode avg rtn : {1}".format(3, avg_return))
