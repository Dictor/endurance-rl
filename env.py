from types import DynamicClassAttribute
from airsimConnector import airsimConnector

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
    def __init__(self, airsimPort, envName):
        # action: front, turn left, turn right
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        # observation: 3 sensors (left, center, right), senser values are divided by 20 levels.
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7, ), dtype=np.float, minimum=0, name='observation')
        self._state = [0, 0, 0, 0, 0, 0, 0]
        self._episode_ended = False
        self.connector = airsimConnector(airsimPort, envName)
        self.connector.connect()
        self.connector.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0, 0, 0, 0, 0, 0, 0]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            self.connector.reset()
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 0:
            self.connector.moveForward()
        elif action == 1:
            self.connector.turnLeft()
        elif action == 2:
            self.connector.turnRight()
        else:
            raise ValueError('action value should be 0 ~ 2.')

        bright = self.connector.getBright()
        distance = self.connector.getDistance()
        goalDistance = self.connector.getGoalDistance()
        self._state = [(bright[0] / 255) * 600, (bright[1] / 255) * 600,
                       (bright[2] / 255) * 600, distance[0], distance[1], distance[2], distance[3]]

        if goalDistance < 8:
            # found goal
            self._episode_ended = True
            print("[EnduranceEnv] termination: goal reached")
            return ts.termination(np.array(self._state, dtype=np.float), 20)
        else:
            # not found goal
            if self.connector.isCollided():
                self._episode_ended = True
                print("[EnduranceEnv] termination: colided")
                return ts.termination(np.array(self._state, dtype=np.float), -20)

            r = -0.05
            if goalDistance > 30:
                r -= (goalDistance - 30) / 10
            else:
                r += (30 - goalDistance)
            print("[EnduranceEnv] transition: reward={:.3f}".format(r))
            return ts.transition(np.array(self._state, dtype=np.float), reward=r, discount=1)
