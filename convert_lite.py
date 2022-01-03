import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import policy_saver

from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks import q_network

policy_dir = "policy"
converter = tf.lite.TFLiteConverter.from_saved_model(
    policy_dir, signature_keys=["action"])
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_policy = converter.convert()
with open('policy.tflite', 'wb') as f:
    f.write(tflite_policy)
