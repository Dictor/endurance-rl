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
            shape=(), dtype=np.int, minimum=0, maximum=2, name='action')
        # observation: 3 sensors (left, center, right), senser values are divided by 20 levels.
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7, ), dtype=np.int, minimum=0, maximum=10, name='observation')
        self._state = [0, 0, 0, 0, 0, 0, 0]
        self._episode_ended = False
        self.connector = airsimConnector(airsimPort, envName)
        self.connector.connect()
        self.connector.reset()
        self.step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0, 0, 0, 0, 0, 0, 0]
        self.connector.reset()
        self._episode_ended = False
        self.step_count = 0
        return ts.restart(np.array(self._state, dtype=np.int))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            self.connector.reset()
            return self.reset()

        # Make sure episodes don't go on forever.
        self.step_count += 1
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
        self._state = [bright[0], bright[1], bright[2],
                       distance[0], distance[1], distance[2], distance[3]]

        if goalDistance < 8:
            # found goal
            self._episode_ended = True
            print("[EnduranceEnv] termination: goal reached")
            return ts.termination(np.array(self._state, dtype=np.int), 50)
        else:
            # not found goal
            if self.connector.isCollided():
                self._episode_ended = True
                print("[EnduranceEnv] termination: colided")
                return ts.termination(np.array(self._state, dtype=np.int), -10)

            if self.step_count > 100:
                self._episode_ended = True
                print("[EnduranceEnv] termination: step limit over")
                return ts.termination(np.array(self._state, dtype=np.int), -3)

            r = -0.05
            if goalDistance > 15:
                r -= (goalDistance - 15) / 1000
            else:
                r += (15 - goalDistance) / 100
            #print("[EnduranceEnv] transition: reward={:.3f}".format(r))
            return ts.transition(np.array(self._state, dtype=np.int), reward=r, discount=0.95)
