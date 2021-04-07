import tensorflow as tf
import numpy as np
import matplotlib as plt
from Tensor import Tensor
from ActionMemory import ActionMemory

import os.path
import pickle

from Ballot import Ballot
from DefaultConfigOptions import *
from ElectionConstructor import ElectionConstructor, construct_irv, construct_h2h
from ModelStats import ModelStats
from NDPopulation import NDPopulation
from ProcessResult import ProcessResult
from Timings import Timings
from PluralityElection import PluralityElection
import datetime as datetime
import tensorflow.keras.optimizers as opt

# Parameters for Ornstein–Uhlenbeck process
THETA = 0.15
DT = 1e-1


class ElectionStatePreprocessor(tf.keras.Model)
    def get_config(self):
        pass

    def __init__(self, ideology_dim: int, n_latent: int, width: int, learn_rate: float):
        super().__init__()
        self.ideology_dim = ideology_dim
        self.n_latent = n_latent

        self.encoding_layers = []
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-enc1"))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-enc2"))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-enc3"))

        self.state = tf.keras.layers.Dense(self.n_latent)


    # input is a tensor of shape (batch_size, n_observations (n_candidates), input_dim)
    def call(self, state: Tensor, training: bool = None, mask: bool = None) -> Tensor:
        # runs the encoder portion of the model on a single input
        if state.shape[1] != 0:
            x = state
            for e in self.encoding_layers:
                x = self.dropout(e(x), training=training)
            # reduce to state observations
            encoded_observations = self.dropout(self.state(x), training=training)
            # now, sum the observations (which have been put on dim 1)
            encoded_state = tf.reduce_sum(encoded_observations, axis=1, keepdims=False)
        else:
            # this corresponds to no candidates in the race yet.
            batch_size = state.shape[0]
            encoded_state = tf.zeros(shape=(batch_size, self.n_latent), dtype=tf.dtypes.float32)

        return encoded_state


class CandidateActor(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, ideology_dim: int, n_latent: int, width: int, learn_rate: float):
        super().__init__()
        self.ideology_dim = ideology_dim
        self.n_latent = n_latent

        self.encoding_layers = []
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-enc1"))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-enc2"))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-enc3"))

        self.state = tf.keras.layers.Dense(self.n_latent)

        self.decoding_layers = []
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-dec1"))
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-dec2"))
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="actor-dec3"))

        self.dropout = tf.keras.layers.Dropout(.3, name="actor-dropout")
        self.returns = tf.keras.layers.Dense(ideology_dim, name="actor-returns")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

    # input is a tensor of shape (batch_size, n_observations (n_candidates), input_dim)
    def call(self, state: Tensor, training: bool = None, mask: bool = None) -> Tensor:
        # runs the encoder portion of the model on a single input
        if state.shape[1] != 0:
            x = state
            for e in self.encoding_layers:
                x = self.dropout(e(x), training=training)
            # reduce to state observations
            encoded_observations = self.dropout(self.state(x), training=training)
            # now, sum the observations (which have been put on dim 1)
            encoded_state = tf.reduce_sum(encoded_observations, axis=1, keepdims=False)
        else:
            # this corresponds to no candidates in the race yet.
            batch_size = state.shape[0]
            encoded_state = tf.zeros(shape=(batch_size, self.n_latent), dtype=tf.dtypes.float32)

        # use that composite state to predict the returns for each possible action
        x = encoded_state
        for d in self.decoding_layers:
            x = self.dropout(d(x), training=training)

        result = tf.tanh(self.returns(x)) * 2
        return result


class CandidateCritic(tf.keras.Model):
    def __init__(self, ideology_dim: int, n_latent: int, width: int, learn_rate: float):
        super().__init__()
        self.ideology_dim = ideology_dim
        self.n_latent = n_latent

        self.encoding_layers = []
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="critc-enc1"))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="critc-enc2"))
        self.encoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="critc-enc3"))

        self.state = tf.keras.layers.Dense(self.n_latent, name="critic_encoded_observations")

        self.decoding_layers = []
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="critic-dec1"))
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="critic-dec2"))
        self.decoding_layers.append(tf.keras.layers.Dense(width, activation='relu', name="critic-dec3"))

        self.dropout = tf.keras.layers.Dropout(.3)
        self.returns = tf.keras.layers.Dense(ideology_dim, name="critic-returns")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)

    # input is a tensor of shape (batch_size, n_observations (n_candidates), input_dim)
    def call(self, state: Tensor, action: Tensor, training: bool = None, mask: bool = None) -> Tensor:
        if state.shape[1] != 0:
            x = state
            for e in self.encoding_layers:
                x = self.dropout(e(x), training=training)
            # reduce to state observations
            encoded_observations = self.dropout(self.state(x), training=training)
            # now, sum the observations (which have been put on dim 1)
            encoded_state = tf.reduce_sum(encoded_observations, axis=1, keepdims=False)
        else:
            # this corresponds to no candidates in the race yet.
            batch_size = state.shape[0]
            encoded_state = tf.zeros(shape=(batch_size, self.n_latent), dtype=tf.dtypes.float32)

        # use the composite state and action to predict the returns for the given action
        x = tf.concat([encoded_state, action], axis=1)
        for d in self.decoding_layers:
            x = self.dropout(d(x), training=training)

        return self.returns(x)


class CandidateAgent:
    def __init__(self, ideology_dim: int, n_latent: int, width: int, actor_lr: float, critic_lr: float):
        self.ideology_dim = ideology_dim
        self.n_latent = n_latent
        self.width = width
        self.gamma = .99
        self.tau = .01

        self.actor = CandidateActor(ideology_dim, n_latent, width, actor_lr)
        self.critic = CandidateCritic(ideology_dim, n_latent, width, critic_lr)
        self.memory = ActionMemory(1024, ideology_dim, ideology_dim)

        self.lower_bound = -2
        self.upper_bound = 2
        self.global_step = 0

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/' + current_time + '/train'
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def train(self, batch_size: int):
        for depth in self.memory.depths():
            state, action, reward = self.memory.get_batch(depth, batch_size)
            self.update(state, action, reward)
        self.global_step += 1

    def update(self, state, action, reward):
        with tf.GradientTape() as tape:
            critic_value = self.critic.call(state, action)
            # print("critic_reward")
            # for i in range(reward.shape[0]):
            #     print(f"\ta: {action.numpy()[i, 0]: 8.2f} ", end="")
            #     print(f" c: {critic_value.numpy()[i, 0]: 5.2f}", end="")
            #     print(f" t: {reward.numpy()[i, 0]:.2f}")

            critic_loss = tf.math.reduce_mean(tf.keras.losses.MSE(reward, critic_value))

        # print(f"critic_loss: {critic_loss.numpy().shape} {critic_loss.numpy(): .2f}")

        with self.summary_writer.as_default():
            label = f"critic_loss-{state.shape[1]}"
            tf.summary.scalar(label, critic_loss, step=self.global_step)
            tf.summary.flush()

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        gv = [(g, v) for g, v in zip(critic_gradient, self.critic.trainable_variables) if g is not None]
        self.critic.optimizer.apply_gradients(gv)

        with tf.GradientTape() as tape:
            policy_actions = self.actor(state)
            actor_loss = -self.critic(state, policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        gv = [(g, v) for g, v in zip(actor_gradient, self.actor.trainable_variables) if g is not None]
        self.actor.optimizer.apply_gradients(gv)

    def _ornstein_uhlenbeck_process(self, x, theta=THETA, mu=0, dt=DT, std=0.2):
        """
        Ornstein–Uhlenbeck process
        """
        return x + theta * (mu - x) * dt + std * np.sqrt(dt) * np.random.normal(size=self.ideology_dim)

    def get_action(self, observation, noise, evaluation=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluation:
            self.noise = self._ornstein_uhlenbeck_process(noise)
            actions += self.noise

        actions = tf.clip_by_value(actions, self.lower_bound, self.upper_bound)

        return actions[0]

    def ready(self) -> bool:
        return self.memory.ready()

    @staticmethod
    def convert_ideology_to_input(ideology: Ideology) -> Tensor:
        return ideology.vec.astype(dtype=np.float32)

    def choose_ideology(self, opponents: List[Candidate]):
        state = self.get_state_from_opponents(opponents)
        ideology_pred = self.actor.call(state, training=True)
        ideology_pred = tf.reshape(ideology_pred, shape=(self.ideology_dim,))
        return ideology_pred.numpy()

    def get_state_from_opponents(self, opponents: List[Candidate]) -> Tensor:
        # shape is (observation_id, ideology_representation_vec)
        if len(opponents) != 0:
            candidate_observations = [self.convert_ideology_to_input(o.ideology) for o in opponents]
            state = np.stack(candidate_observations)
        else:
            state = tf.zeros(shape=(0, self.ideology_dim), dtype=tf.dtypes.float32)

        return tf.expand_dims(state, 0)

    def add_sample_from_candidates(self, candidate: Candidate, opponents: List[Candidate], winner: Candidate):
        state = self.get_state_from_opponents(opponents)

        action = self.convert_ideology_to_input(candidate.ideology)
        action = tf.expand_dims(action, 0)

        if winner == candidate:
            reward = tf.ones(shape=(1, 1), dtype=tf.dtypes.float32)
        else:
            reward = tf.zeros(shape=(1, 1), dtype=tf.dtypes.float32)

        self.memory.add_sample(state, action, reward)

    def save_to_file(self, path: str):
        self.actor.save(path + ".actor")
        self.critic.save(path + ".critic")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the model
        del state["actor"]
        del state["critic"]
        del state["memory"]
        del state["optimizer"]
        del state["summary_writer"]
        return state


def create_model_and_population(ideology_dim: int) -> (CandidateAgent, NDPopulation):
    hidden_ratio = 64
    n_hidden = hidden_ratio * ideology_dim
    n_latent = ideology_dim * 32
    batch_size = 128
    learn_rate = .001

    model = CandidateAgent(ideology_dim=ideology_dim,
                           n_latent=n_latent,
                           width=n_hidden,
                           actor_lr=learn_rate,
                           critic_lr=learn_rate)

    population_means = np.zeros(shape=(ideology_dim,))
    population_stddev = np.ones(shape=(ideology_dim,))
    pop = NDPopulation(population_means, population_stddev)
    return model, pop


def measure_representation(candidate: Candidate, voters: List[Voter]) -> float:
    n_voters = len(voters)
    balance = []
    for d in range(candidate.ideology.dim):
        lc = len([v for v in voters if v.ideology.vec[d] < candidate.ideology.vec[d]])
        balance.append(min(lc / n_voters, 1 - lc / n_voters))
    return float(np.mean(balance))


def gen_non_model_candidates(model: CandidateAgent, population: NDPopulation) -> List[Candidate]:
    candidates: List[Candidate] = []
    if model.ready():
        if np.random.choice([True, False]):
            candidates += gen_pilot_candidates(population, .8)
        else:
            candidates += gen_random_candidates(population, 3)
    else:
        candidates += gen_pilot_candidates(population, .6)
        candidates += gen_random_candidates(population, 3)

    np.random.shuffle(candidates)
    return candidates


def gen_pilot_candidates(population: NDPopulation, spacing: float) -> List[Candidate]:
    candidates = []
    dim = population.dim
    d = spacing
    fuzz = .05
    c1_vec = np.random.normal(0, .01, dim)
    c1_vec[0] += np.random.normal(d, fuzz)
    candidates.append(Candidate("P-R", Independents, ideology=Ideology(c1_vec), quality=0))

    c2_vec = np.random.normal(0, .01, dim)
    c2_vec[0] -= np.random.normal(d, fuzz)
    candidates.append(Candidate("P-L", Independents, ideology=Ideology(c2_vec), quality=0))

    c3_vec = np.random.normal(0, .02, dim)
    candidates.append(Candidate("P-C", Independents, ideology=Ideology(c3_vec), quality=0))

    return candidates


def gen_random_candidates(population: NDPopulation, n: int) -> List[Candidate]:
    candidates = []
    for i in range(3):
        ivec = population.unit_sample_voter().ideology.vec * .5
        candidates.append(Candidate("r-" + str(i), Independents, Ideology(ivec), 0))

    return candidates


def run_sample_election(model: CandidateAgent, process: ElectionConstructor, population: NDPopulation, train: bool):
    candidates = []
    model_entries = set(np.random.choice(range(6), 3, replace=False))
    r_candidates = gen_non_model_candidates(model, population)
    for i in range(6):
        if i in model_entries and model.ready():
            ideology = Ideology(model.choose_ideology(candidates))
            c = Candidate("m-" + str(i), Independents, ideology, 0)
        else:
            if train:
                c = r_candidates.pop()
            else:
                ideology = population.unit_sample_voter().ideology
                c = Candidate("r-" + str(i), Independents, ideology, 0)

        candidates += [c]

    voters = population.generate_unit_voters(1000)
    ballots = [Ballot(v, candidates, unit_election_config) for v in voters]
    result = process.run(ballots, set(candidates))
    winner = result.winner()
    balance = measure_representation(winner, voters)

    return winner, candidates, balance


def train_candidate_model(model: CandidateAgent, process: ElectionConstructor, population: NDPopulation,
                          max_steps=5000):
    timings = Timings()
    stats = ModelStats()
    first = True
    while model.global_step < max_steps:
        winner, candidates, balance = run_sample_election(model, process, population, True)
        for i in range(len(candidates)):
            model.add_sample_from_candidates(candidates[i], candidates[0:i], winner)

        if model.ready():
            if first:
                print("starting to train")
                first = False

            stats.update(winner, candidates, balance)
            with timings.time_block("model.train"):
                model.train(128)
            s = model.global_step
            if (s < 100 and s % 10 == 0) or (s < 1000 and s % 100 == 0) or s % 1000 == 0:
                stats.print(process.name, model.global_step)
                stats.reset()

    timings.print()


def check_stats(stats: ModelStats, model: CandidateAgent, process: ElectionConstructor, population: NDPopulation):
    results = []
    timings = Timings()
    for i in range(1000):
        winner, candidates, balance = run_sample_election(model, process, population, train=False)
        stats.update(winner, candidates, balance)


def run_parameter_set(process: ElectionConstructor, ibins: int, dim: int):
    save_path = "models/cm-%s-%03d-%dD.p" % (process.name, ibins, dim)
    model, population = create_model_and_population(dim)
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            model: CandidateAgent = pickle.load(f)
    else:
        train_candidate_model(model, process, population)
        # Saving the model file is not working at this time.
        model.save_to_file(save_path)

    stats = ModelStats()
    check_stats(stats, model, process, population)
    return stats, model


def train_models():
    dims = [1]
    processes = [
        ElectionConstructor(constructor=construct_irv, name="Instant Runoff"),
        ElectionConstructor(constructor=construct_h2h, name="Head-to-Head"),
    ]

    results = []
    for bins in [64, 128]:
        for process in processes:
            for dim in dims:
                stats, model = run_parameter_set(process, bins, dim)
                results.append(ProcessResult(process, bins, dim, stats, 10000))
                results[-1].print()

    for r in results:
        r.print()


def save_test():
    process = ElectionConstructor(constructor=construct_irv, name="Instant Runoff")
    model, population = create_model_and_population(ideology_dim=1)

    train_candidate_model(model, process, population, 500000)


save_test()
