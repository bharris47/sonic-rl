import random

import gym
import retro


class PlaylistEnv(gym.Wrapper):
    def __init__(self, env_args):
        self._env_args = env_args
        self.current_env_args = random.choice(self._env_args)
        self.env = retro.make(**self.current_env_args)
        super(PlaylistEnv, self).__init__(self.env)

    def reset(self, **kwargs):
        if self.env:
            self.env.close()
        self.current_env_args = random.choice(self._env_args)
        self.env = retro.make(**self.current_env_args)
        return self.env.reset(**kwargs)

    @property
    def current_game(self):
        if self.current_env_args:
            return self.current_env_args['game']
        return None

    @property
    def current_state(self):
        if self.current_env_args:
            return self.current_env_args['state']
        return None

    @property
    def data(self):
        return self.env.data