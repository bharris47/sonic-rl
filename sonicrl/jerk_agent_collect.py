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
import numpy as np
import retro

EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(1e6)
BACKTRACK_REWARD_THRESHOLD = 10000

class ObservationSaver:
    def __init__(self, output_path, frame_format, image_directory):
        self._outfile = open(output_path, 'w')
        self._frame_format = frame_format
        self._image_directory = image_directory

    def save(self, observation, action, reward, done):
        uuid = uuid4().hex
        frame_name = self._frame_format.format(uuid=uuid)
        frame_path = os.path.join(args.image_directory, frame_name)
        line = dict(image_id=frame_name, reward=reward, action=action.tolist(), done=done)
        self._outfile.write(json.dumps(line) + '\n')

        bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, bgr)

    def __del__(self):
        self._outfile.close()


def main(env, saver):
    """Run JERK on the attached environment."""
    new_ep = True
    observation_buffer = []
    while True:
        if new_ep:
            _ = env.reset()
        reward, new_ep = move(env, 100, observation_buffer)
        if not new_ep and reward <= BACKTRACK_REWARD_THRESHOLD:
            print('backtracking due to negative reward: %f' % reward)
            reward, new_ep = move(env, 70, observation_buffer, left=True)
            print('received reward after backtracking: %f' % reward)
        else:
            print('received reward: %f' % reward)

        while len(observation_buffer[0]) == 4:
            obs, reward, done, action = observation_buffer.pop(0)
            saver.save(obs, action, reward, done)


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
        env.render(mode='human')
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
        env.render(mode='human')
        idx += 1
    return env.total_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help='the name or path for the game to run')
    parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
    parser.add_argument('output_file', help='file to store metadata')
    parser.add_argument('image_directory', help='directory in which to store frames')
    parser.add_argument('--scenario', '-s', default='scenario',
                        help='the scenario file to load, minus the extension')
    parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='increase verbosity (can be specified multiple times)')
    parser.add_argument('--quiet', '-q', action='count', default=0,
                        help='decrease verbosity (can be specified multiple times)')
    args = parser.parse_args()

    env = retro.make(args.game, args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)

    frame_format = '{game}_{state}_{{uuid}}.jpg'.format(game=args.game, state=args.state)
    saver = ObservationSaver(args.output_file, frame_format, args.image_directory)

    main(env, saver)
