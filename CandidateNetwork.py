import tensorflow as tf
import numpy as np
from typing import List
from Tensor import Tensor



class CandidateNetwork(tf.keras.Model):
    def __init__(self, ideology_bins: int, ideology_dim: int, n_latent: int, width: int):
        super().__init__()
        self.ideology_bins = ideology_bins
        self.ideology_dim = ideology_dim
        self.n_latent = n_latent

        self.encoding_layers = []
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu'))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu'))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu'))

        self.state = tf.keras.layers.Dense(self.n_latent)

        self.decoding_layers = []
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu'))
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu'))
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu'))

        self.returns = tf.keras.layers.Dense(ideology_bins * ideology_dim)

    # input is a tensor of shape (batch_size, n_observations (n_candidates), input_dim)
    def call(self, input: Tensor) -> Tensor:
        # runs the encoder portion of the model on a single input
        if input.shape[1] != 0:
            x = input
            for e in self.encoding_layers:
                x = e(x)
            # reduce to state observations
            encoded_observations = self.state(x)
            # now, sum the observations (which have been put on dim 1)
            encoded_state = tf.reduce_sum(encoded_observations, axis=1, keepdims=False)
        else:
            # this corresponds to no candidates in the race yet.
            batch_size = input.shape[0]
            encoded_state = tf.zeros(shape=(batch_size, self.n_latent), dtype=tf.dtypes.float32)

        # use that composite state to predict the returns for each possible action
        x = encoded_state
        for d in self.decoding_layers:
            x = d(x)

        return self.returns(x)