import argparse
import os.path
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import matplotlib.pyplot as plt

from Ballot import Ballot
from CandidateModel import CandidateModel
from DefaultConfigOptions import *
from NDPopulation import NDPopulation
from Timings import Timings
from ElectionConstructor import ElectionConstructor, construct_irv, construct_h2h
from ModelStats import ModelStats
from ProcessResult import ProcessResult


class Sample:
    def __init__(self, opponents: List[Candidate], candidate: Candidate):
        self.opponents = opponents.copy()
        self.candidate = candidate


def create_model_and_population(ideology_bins: int, ideology_dim: int) -> (CandidateModel, NDPopulation):
    ideology_bins = 64
    hidden_ratio = 4
    n_hidden = hidden_ratio * ideology_bins * ideology_dim
    n_latent = ideology_bins * ideology_dim
    batch_size = 128
    learn_rate = .001

    model = CandidateModel(ideology_bins=ideology_bins,
                           ideology_dim=ideology_dim,
                           n_hidden=n_hidden,
                           n_latent=n_latent,
                           learn_rate=learn_rate)

    population_means = np.zeros(shape=(ideology_dim,))
    population_stddev = np.ones(shape=(ideology_dim,))
    pop = NDPopulation(population_means, population_stddev)
    return model, pop
#%%

def gen_non_model_candidates(model: CandidateModel, population: NDPopulation) -> List[Candidate]:
    candidates: List[Candidate] = []
    if model.ready():
        if np.random.choice([True, False]):
            candidates += gen_example_candidates(population, .7)
        else:
            candidates += gen_random_candidates(population, 3)
    else:
        candidates += gen_example_candidates(population, .6)
        candidates += gen_random_candidates(population, 3)

    np.random.shuffle(candidates)
    return candidates

def gen_example_candidates(population: NDPopulation, spacing: float) -> List[Candidate]:
    candidates = []
    dim = population.dim
    d = spacing
    fuzz = .05
    c1_vec = np.random.normal(0, .01, dim)
    c1_vec[0] += np.random.normal(d, fuzz)
    candidates.append( Candidate("P-R", Independents, ideology=Ideology(c1_vec), quality=0))

    c2_vec = np.random.normal(0, .01, dim)
    c2_vec[0] -= np.random.normal(d, fuzz)
    candidates.append(Candidate("P-L", Independents, ideology=Ideology(c2_vec), quality=0))

    c3_vec = np.random.normal(0, .01, dim)
    candidates.append(Candidate("P-C", Independents, ideology=Ideology(c3_vec), quality=0))

    return candidates

def gen_random_candidates(population: NDPopulation, n: int)-> List[Candidate]:
    candidates = []
    for i in range(3):
        ivec = population.unit_sample_voter().ideology.vec * .5
        candidates.append(Candidate("r-" + str(i), Independents, Ideology(ivec), 0))

    return candidates



def run_sample_election(model: CandidateModel, process: ElectionConstructor, population: NDPopulation, train: bool):
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
    #result = process.run(ballots, set(candidates))
    election = process.constructor(ballots, set(candidates))
    result = election.result()
    min_distance = min([c.ideology.distance_from_o() for c in candidates])
    winner = result.winner()

    delta = winner.ideology.distance_from_o() - min_distance
    if delta > .10:
        print("bad winner: %.6f" % delta)
        for c in candidates:
            print("%s %.3f" % (c.name, c.ideology.distance_from_o()), end = "")
            if (c == winner):
                print(" winner", end = "")
            print("")
        print("")

    balance = 0

    return winner, candidates, balance


def train_candidate_model(model: CandidateModel, process: ElectionConstructor, population: NDPopulation, max_steps: int):
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
            for i in range(5):
                with timings.time_block("model.train"):
                    model.train(128)

            if model.global_step % 1000 == 0:
                stats.print(process.name, model.global_step)
                if model.global_step < max_steps:
                    stats.reset()

    timings.print()


def check_stats(stats: ModelStats, model: CandidateModel, process: ElectionConstructor, population: NDPopulation):
    results = []
    timings = Timings()
    for i in range(1000):
        winner, candidates, balance = run_sample_election(model, process, population, train=False)
        stats.update(winner, candidates, balance)
        results.append((winner, candidates))


def run_parameter_set(process: ElectionConstructor, ibins: int, dim: int, steps: int) -> ProcessResult:
    save_path = "models/cm-%s-%03d-%dD.p" % (process.name, ibins, dim)
    model, population = create_model_and_population(ibins, dim)
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            model: CandidateModel = pickle.load(f)
    else:
        train_candidate_model(model, process, population, steps)
        # Saving the model file is not working at this time.
        # model.save_to_file(save_path)

    stats = ModelStats()
    check_stats(stats, model, process, population)
    return ProcessResult(process, ibins, dim, stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", help="dimensionality", type=int, default=1)
    parser.add_argument("--bins", help="ideology bins", type=int, default=64)
    parser.add_argument("--steps", help="learning steps", type=int, default=5000)
    parser.add_argument("--process", help="election proces: Hare or Minimax", type=str, default="Minimax")
    parser.add_argument("--output", help="Location for output", type=str)
    args = parser.parse_args()
    print("dim: ", args.dim)
    print("bins: ", args.bins)
    print("process: ", args.process)
    print("output: ", args.output)

    if args.process == "Hare":
        process = ElectionConstructor(construct_irv, "Hare")
    else:
        process = ElectionConstructor(construct_h2h, "Minimax")

    result = run_parameter_set(process, args.bins, args.dim, args.steps)
    result.save(args.output)