import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn, mlp_extractor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
import numpy as np


# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class LstmCustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 net_arch=None,
                 act_fun=tf.tanh, feature_extraction="mlp", **kwargs):
        super(LstmCustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

            # extracted_features = tf.keras.layers.Dense(128, activation='relu')(self.processed_obs)
            # extracted_features = tf.keras.layers.MaxPooling1D(pool_size=2)(extracted_features)
            # extracted_features = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same')(extracted_features)
            # extracted_features = tf.keras.layers.MaxPooling1D(pool_size=2)(extracted_features)

            # lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=128)
            # extracted_features = nature_cnn(self.processed_obs, **kwargs)

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:
            if layers is None:
                layers = [128, 64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            # embeded = tf.keras.layers.LSTM(64)(self.processed_obs[1])
            with_energy = tf.concat(( tf.layers.flatten(self.processed_obs[2]), tf.layers.flatten(self.processed_obs[0]), self.processed_obs[3]), axis=-1)
            pi_latent, vf_latent = mlp_extractor(with_energy, net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):

        feed_dict = {x: obs[key] for (i, x), key in zip(enumerate(self.obs_ph), obs.keys())}

        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   feed_dict)
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   feed_dict)
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        feed_dict = {x: obs[key] for (i, x), key in zip(enumerate(self.obs_ph), obs.keys())}
        return self.sess.run(self.value_flat, feed_dict)
