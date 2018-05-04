import numpy as np
import retro
from keras import Input
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import Model
from rl.random import OrnsteinUhlenbeckProcess

MAX_EPISODE_STEPS = 10000

ENV_NAME = 'SonicTheHedgehog-Genesis'
STATE_NAME = 'GreenHillZone.Act1'

EPISODES = 1000


class RetroProcessor(Processor):
    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        shape = batch.shape
        batch = np.reshape(batch, (shape[0] * shape[1],) + shape[2:])
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_action(self, action):
        return np.where(action >= 0.5, 1, 0)


def conv_block(x, filters=32, kernel_size=(3, 3), strides=(4, 4), pool=(2, 2)):
    features = Conv2D(filters, kernel_size=kernel_size, strides=strides, activation='relu')(x)
    features = MaxPool2D(pool_size=pool)(features)
    return features


def actor_model(observeration_shape, n_actions):
    image = Input(shape=observeration_shape)
    features = conv_block(image)
    features = conv_block(features)
    features = Flatten()(features)
    features = Dense(128, activation='relu')(features)
    output = Dense(n_actions, activation='sigmoid')(features)
    return Model(image, features), Model(image, output)


def critic_model(feature_extractor, observeration_shape, n_actions):
    action = Input(shape=(n_actions,))
    action_features = Dense(128, activation='relu')(action)

    image = Input(shape=observeration_shape)
    image_features = feature_extractor(image)

    features = Concatenate()([action_features, image_features])
    features = Dense(64, activation='relu')(features)
    output = Dense(1)(features)
    return action, Model([action, image], output)


if __name__ == '__main__':
    env = retro.make(ENV_NAME, STATE_NAME)

    n_actions = env.action_space.n

    feature_extractor, actor = actor_model(env.observation_space.shape, n_actions)
    actor.summary()

    action_input, critic = critic_model(feature_extractor, env.observation_space.shape, n_actions)
    critic.summary()

    memory = SequentialMemory(limit=MAX_EPISODE_STEPS, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=n_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=10000, nb_steps_warmup_actor=10000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3, processor=RetroProcessor(),
                      train_interval=300)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    agent.fit(env, nb_steps=MAX_EPISODE_STEPS * EPISODES, nb_max_episode_steps=MAX_EPISODE_STEPS,
              visualize=True, verbose=2)
    agent.save_weights('ddpg_{}_weights.h5'.format(ENV_NAME), overwrite=True)
