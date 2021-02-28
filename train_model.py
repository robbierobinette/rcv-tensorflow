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


def run_sample_election(model: CandidateModel, process: ElectionConstructor, population: NDPopulation,
                        timings: Timings):
    candidates = []
    model_entries = set(np.random.choice(range(6), 3, replace=False))
    for i in range(6):
        if i in model_entries and model.ready():
            ideology = Ideology(model.choose_ideology(candidates))
            c = Candidate("m-" + str(i), Independents, ideology, 0)
        else:
            ideology = population.unit_sample_voter().ideology
            c = Candidate("r-" + str(i), Independents, ideology, 0)

        candidates += [c]

    voters = population.generate_unit_voters(1000)
    ballots = [Ballot(v, candidates, unit_election_config) for v in voters]
    result = process.run(ballots, set(candidates))
    winner = result.winner()
    return winner, candidates


def train_candidate_model(model: CandidateModel, process: ElectionConstructor, population: NDPopulation):
    timings = Timings()
    stats = ModelStats()
    first = True
    steps= 10000
    while model.global_step < steps:
        with timings.time_block("run_election"):
            winner, candidates = run_sample_election(model, process, population, timings)
        with timings.time_block("add_sample"):
            for i in range(len(candidates)):
                model.add_sample_from_candidates(candidates[i], candidates[0:i], winner)

        if model.ready():
            if first:
                print("starting to train")
                first = False

            stats.update(winner, candidates)
            with timings.time_block("model.train"):
                model.train(128)
            s = model.global_step
            if s % 1000 == 0:
                stats.print(process.name, model.global_step)
                if model.global_step < steps:
                    stats.reset()

    timings.print()


def check_stats(stats: ModelStats, model: CandidateModel, process: ElectionConstructor, population: NDPopulation):
    results = []
    timings = Timings()
    for i in range(1000):
        winner, candidates = run_sample_election(model, process, population, timings)
        stats.update(winner, candidates)
        results.append((winner, candidates))


def run_parameter_set(process: ElectionConstructor, ibins: int, dim: int) -> ProcessResult:
    save_path = "models/cm-%s-%03d-%dD.p" % (process.name, ibins, dim)
    model, population = create_model_and_population(ibins, dim)
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            model: CandidateModel = pickle.load(f)
    else:
        train_candidate_model(model, process, population)
        # Saving the model file is not working at this time.
        # model.save_to_file(save_path)

    stats = ModelStats()
    check_stats(stats, model, process, population)
    return ProcessResult(process, ibins, dim, stats)


def build_all_models() -> List[ProcessResult]:
    dims = [1, 2, 3, 4]
    processes = [
        ElectionConstructor(constructor=construct_irv, name="Hare"),
        ElectionConstructor(constructor=construct_h2h, name="Minimax")
    ]

    results: List[ProcessResult] = []
    bins = 64

    pairs = []

    def run(process_dim: Tuple[ElectionConstructor, int]) -> ProcessResult:
        proces, dim = process_dim
        return run_parameter_set(process, bins, dim)

    for process in processes:
        for dim in dims:
            pairs.append((process, dim))

    with ThreadPoolExecutor() as executor:
        futures = executor.map(run, pairs)

    results = [f for f in futures]
    return results


def plot_results(results: List[ProcessResult]):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    fig.suptitle("Distance From Origin for Winner With Strategic Candidates", color="black", fontsize=22)

    count = 0
    irv_results = [r for r in results if r.process.name == "Hare"]
    h2h_results = [r for r in results if r.process.name == "Minimax"]

    for ir, hr in zip(irv_results, h2h_results):
        assert (ir.dim == hr.dim)
        row = count // 2
        col = count % 2
        count += 1

        axis = axes[row][col]
        axis.tick_params(axis='x', colors="black")
        axis.tick_params(axis='y', colors="black")
        axis.set_xlim([0, 2.5])

        iv = [w.ideology.distance_from_o() for w in ir.stats.results]
        hv = [w.ideology.distance_from_o() for w in hr.stats.results]

        axis.hist([iv, hv], bins=30, label=[ir.process.name, hr.process.name])
        axis.set_title("Dimensionality: %d" % ir.dim, color="black")

        axis.legend()

    plt.savefig("strategic_candidates.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", help="dimensionality", type=int, default=1)
    parser.add_argument("--bins", help="ideology bins", type=int, default=64)
    parser.add_argument("--process", help="election proces: Hare or Minimax", type=str)
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

    result = run_parameter_set(process, args.bins, args.dim)
    result.save(args.output)