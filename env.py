import airsimConnector as connector

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class EnduranceEnv(py_environment.PyEnvironment):
  def __init__(self):
    # action: front, turn left, turn right
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    # observation: 3 sensors (left, center, right), senser values are divided by 20 levels.
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(3,), dtype=np.int32, minimum=0, maximum=20, name='observation')
    self._state = 0
    self._episode_ended = False
    connector.connect()
    connector.reset()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      connector.reset()
      return self.reset()

    print("action: {0}".format(action))

    # Make sure episodes don't go on forever.
    if action == 0:
      connector.moveForward()
      self._state = action
    elif action == 1:
      connector.turnLeft()
      self._state = action
    elif action == 2:
      connector.turnRight()
      self._state = action
    else:
      raise ValueError('action value should be 0 ~ 2.')

    bright = connector.getBright()
    bright = [bright[0] / 255, bright[1] / 255, bright[0] / 255]
    goalDistance = connector.getGoalDistance()

    if self._episode_ended or goalDistance < 1:
      # found goal
      self._episode_ended = True
      return ts.termination(bright, reward=10)
    else:
      # not found goal
      return ts.transition(bright, reward=-0.3, discount=1)