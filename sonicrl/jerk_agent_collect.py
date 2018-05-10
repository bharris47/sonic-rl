#!/usr/bin/env python

"""
A scripted agent called "Just Enough Retained Knowledge".
"""
import argparse
import json
import os
import random
from uuid import uuid4

import cv2
import gym
import numpy as np

from sonicrl.environments import get_environments
from sonicrl.playlist_env import PlaylistEnv

EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(1e6)
BACKTRACK_REWARD_THRESHOLD = 0


class ObservationSaver:
    def __init__(self, output_path, frame_format, image_directory):
        self._outfile = open(output_path, 'w')
        self._frame_format = frame_format
        self._image_directory = image_directory
        self.num_saved = 0

    def save(self, game, state, observation, action, reward, done):
        uuid = uuid4().hex
        frame_name = self._frame_format.format(game=game, state=state, uuid=uuid)
        frame_path = os.path.join(args.image_directory, frame_name)
        line = dict(game=game, state=state, image_id=frame_name, reward=reward,
                    action=action.tolist(), done=done)
        self._outfile.write(json.dumps(line) + '\n')

        bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, bgr)
        self.num_saved += 1

    def __del__(self):
        self._outfile.close()


def main(env, saver):
    """Run JERK on the attached environment."""
    new_ep = True
    observation_buffer = []
    playlist_env = env.env
    while True:
        if new_ep:
            _ = env.reset()
        reward, new_ep = move(env, 100, observation_buffer)
        if not new_ep and reward <= BACKTRACK_REWARD_THRESHOLD:
            reward, new_ep = move(env, 70, observation_buffer, left=True)

        while len(observation_buffer[0]) == 4:
            obs, reward, done, action = observation_buffer.pop(0)
            saver.save(playlist_env.current_game, playlist_env.current_state, obs, action, reward, done)
            if saver.num_saved % 10000 == 0:
                print('%d observations saved' % saver.num_saved)


def move(env, num_steps, buffer, left=False, jump_prob=1.0 / 40.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        if buffer:
            buffer[-1].append(action)
        obs, rew, done, _ = env.step(action)
        # env.render(mode='human')
        buffer.append([obs, rew, done])
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done


def exploit(env, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done:
        if idx >= len(sequence):
            action = np.zeros((12,), dtype='bool')
        else:
            action = sequence[idx]
        _, _, done, _ = env.step(action)
        # env.render(mode='human')
        idx += 1
    return env.total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help='file to store metadata')
    parser.add_argument('image_directory', help='directory in which to store frames')
    parser.add_argument('--environments', nargs='+',
                        help='one or more environment csvs describing environment game and states (e.g. train/val)')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='increase verbosity (can be specified multiple times)')
    parser.add_argument('--quiet', '-q', action='count', default=0,
                        help='decrease verbosity (can be specified multiple times)')
    args = parser.parse_args()

    environments = []
    for env_file in args.environments:
        environments.extend(get_environments(env_file))
    env = PlaylistEnv(environments)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=18000)

    frame_format = '{game}_{state}_{uuid}.jpg'
    saver = ObservationSaver(args.output_file, frame_format, args.image_directory)
    main(env, saver)
