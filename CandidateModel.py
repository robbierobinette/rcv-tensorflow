from CandidateNetwork import CandidateNetwork
from Candidate import Candidate
import numpy as np
import tensorflow as tf
from typing import List
from ActionMemory import ActionMemory
from Ideology import Ideology
import random
import datetime as datetime
from Timings import Timings

from Tensor import Tensor


class CandidateModel:
    def __init__(self, input_dim: int, ideology_width: int, ideology_dim: int, n_hidden: int, batch_size: int,
                 learn_rate: int):
        super().__init__()
        self.ideology_width = ideology_width
        self.ideology_dim = ideology_dim
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.model = CandidateNetwork(ideology_width=ideology_width, ideology_dim=ideology_dim, state_dim=30,
                                      width=n_hidden)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        self.global_step = 0
        self.timings = Timings()
        self.memory = ActionMemory(100 * 1000, ideology_dim, ideology_dim, self.timings)
        self.action_dim = self.ideology_width * self.ideology_dim
        # this is the dimension of the input vector for a single opponent.  It can be the same as ideology_dim, or
        # it could be ideology_dim * ideology_width for a one_hot representation of ideology
        self.input_dim = input_dim

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/' + current_time + '/train'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def ready(self) -> bool:
        return self.memory.ready()

    def train(self, batch_size: int):
        for depth in self.memory.depths():
            state, action, reward = self.memory.get_batch(depth, batch_size)
            self.update(state, action, reward)

    def update(self, input_batch: np.ndarray, actions: np.ndarray, reward: np.ndarray):
        batch_size = input_batch.shape[0]
        one_hot = tf.one_hot(actions, depth=self.ideology_width)
        # flatten the one_hot array out to match the output of the network
        # each row will have 'ideology_dim' hot elements, one in each chunk of 'ideology_width'
        one_hot = tf.reshape(one_hot, shape=(batch_size, self.action_dim))
        with tf.GradientTape() as tape:
            y = self.model(input_batch)
            rewards = tf.ones(shape=(batch_size, self.action_dim)) * reward
            deltas = tf.square(y - rewards) * one_hot
            loss = tf.reduce_sum(deltas)

        with self.summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=self.global_step)

        grads = tape.gradient(loss, self.model.variables)
        self.optimizer.apply_gradients(zip(grads, self.model.variables), self.global_step)
        self.global_step += 1
        if self.global_step % 100 == 0:
            self.timings.print()
            self.timings = Timings()

    def convert_ideology_to_input(self, ideology: Ideology) -> Tensor:
        return ideology.vec.astype(dtype=np.float32)

    def convert_ideology_to_int(self, ideology: float):
        return int((ideology + 100) / 200 * self.ideology_width)

    # the action vector is a vector of integers corresponding to the actions
    # taken where each action is a location on the i'th dimension of the
    # ideological spectrum
    # i.e.  an ideology of [0,0,0] would correspond to [100, 100, 100]
    #
    def convert_ideology_to_action_vec(self, ideology: Ideology) -> Tensor:
        ii = [self.convert_ideology_to_int(i) for i in ideology.vec]
        return tf.constant(ii, dtype=tf.dtypes.int32)

    def get_state_from_opponents(self, opponents: List[Candidate]) -> Tensor:
        # shape is (observation_id, ideology_representation_vec)
        if len(opponents) != 0:
            candidate_observations = [self.convert_ideology_to_input(o.ideology) for o in opponents]
            state = np.stack(candidate_observations)
        else:
            state = tf.zeros(shape=(0, self.input_dim), dtype=tf.dtypes.float32)

        return tf.expand_dims(state, 0)

    def add_sample_from_candidates(self, candidate: Candidate, opponents: List[Candidate], winner: Candidate):
        state = self.get_state_from_opponents(opponents)

        action = self.convert_ideology_to_action_vec(candidate.ideology)
        action = tf.expand_dims(action, 0)

        if winner == candidate:
            reward = tf.ones(shape=(1, 1), dtype=tf.dtypes.float32)
        else:
            reward = tf.zeros(shape=(1, 1), dtype=tf.dtypes.float32)

        self.memory.add_sample(state, action, reward)

    def choose_ideology(self, opponents: List[Candidate]) -> Tensor:
        state = self.get_state_from_opponents(opponents)
        ideology_pred = self.model.call(state)
        ideology_hot = tf.reshape(ideology_pred, shape=(self.ideology_dim, self.ideology_width))
        ideology_indices = tf.cast(tf.argmax(ideology_hot, axis=1), tf.dtypes.float32)
        ideology_vec = ideology_indices * 200.0 / self.ideology_width + 100.0
        return ideology_vec.numpy()








