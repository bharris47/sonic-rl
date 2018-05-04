import numpy as np
import retro
from keras import Input
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import Model, EpsGreedyQPolicy, LinearAnnealedPolicy

MAX_EPISODE_STEPS = 10000

ENV_NAME = 'SonicTheHedgehog-Genesis'
STATE_NAME = 'GreenHillZone.Act1'

EPISODES = 1000


class RetroProcessor(Processor):
    def __init__(self, n_actions):
        self._n_actions = n_actions

    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        shape = batch.shape
        batch = np.reshape(batch, (shape[0] * shape[1],) + shape[2:])
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_action(self, action):
        action_vector = np.zeros(self._n_actions)
        action_vector[action] = 1
        return action_vector


def conv_net(input_shape, n_actions):
    image = Input(shape=input_shape)

    features = Conv2D(32, (3, 3), strides=(4, 4), activation='relu')(image)
    features = Conv2D(64, (3, 3), strides=(4, 4), activation='relu')(features)
    features = Flatten()(features)
    features = Dense(128, activation='relu')(features)
    output = Dense(n_actions)(features)

    return Model(image, output)


if __name__ == '__main__':
    env = retro.make(ENV_NAME, STATE_NAME)

    observation = env.reset()

    n_actions = env.action_space.n

    model = conv_net(observation.shape, n_actions)
    model.summary()

    memory = SequentialMemory(limit=MAX_EPISODE_STEPS, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=100000)
    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, target_model_update=1e-2, policy=policy,
                   processor=RetroProcessor(n_actions), train_interval=300)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=MAX_EPISODE_STEPS * EPISODES, nb_max_episode_steps=MAX_EPISODE_STEPS,
            visualize=True, verbose=2)
    dqn.save_weights('dqn_{}_weights.h5'.format(ENV_NAME), overwrite=True)
