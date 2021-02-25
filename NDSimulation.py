import matplotlib.pyplot as plt

from Ballot import Ballot
from DefaultConfigOptions import *
from ElectionResult import ElectionResult
from InstantRunoffElection import InstantRunoffElection
from HeadToHeadElection import HeadToHeadElection
from Population import Population
from NDPopulation import NDPopulation
from typing import List, Set, Callable
from Election import Election

all_voters = np.empty(dtype=float, shape=0)
all_candidates = np.empty(dtype=float, shape=0)


class ElectionConstructor:
    def __init__(self, constructor: Callable[[List[Ballot], Set[Candidate]], Election], name: str):
        self.constructor = constructor
        self.name = name

    def run(self, ballots: List[Ballot], candidates: Set[Candidate]) -> ElectionResult:
        e = self.constructor(ballots, candidates)
        return e.result()


def construct_irv(ballots: List[Ballot], candidates: Set[Candidate]):
    return InstantRunoffElection(ballots, candidates)


def construct_h2h(ballots: List[Ballot], candidates: Set[Candidate]):
    return HeadToHeadElection(ballots, candidates)


def main():
    winners: List[List[ElectionResult]] = []
    processes = [
        ElectionConstructor(construct_irv, "Instant Runoff"),
        ElectionConstructor(construct_h2h, "Head to Head")
    ]

    for i in range(1):
        print("iteration %d" % i)
        c = 0
        for ii in range(1000):
            winners.append(run_election(processes))
            c += 1
            print(".", end="")
            if c % 100 == 0:
                print("")
        print("")

    for process_index in range(len(processes)):
        d = get_plot_column(winners, process_index, Independents)
        print("mean distance of winner from center %.2f" % (sum(d) / len(d)))

        plt.hist([d],
                 stacked=True,
                 density=True,
                 bins=30,
                 color=["purple"],
                 label=["Winners"],
                 )
        plt.title(processes[process_index].name)
        plt.xlabel('distance from center')
        plt.ylabel('count')
        plt.show()

    plt.hist([all_voters],
             stacked=True,
             density=True,
             bins=30,
             color=["purple"],
             label=["Voters"],
             )
    plt.title("Voters")
    plt.xlabel('distance from median')
    plt.ylabel('count')
    plt.show()

    plt.hist([all_candidates],
             stacked=True,
             density=True,
             bins=30,
             color=["purple"],
             label=["Candidates"],
             )
    plt.title("Candidates")
    plt.xlabel('distance from median')
    plt.ylabel('count')
    plt.show()


def get_plot_column(winners: List[List[ElectionResult]], process_index: int, party: Party) -> List[float]:
    ideologies = [r[process_index].winner().ideology for r in winners if r[process_index].winner().party == party]
    distances = [i.distance_from_o() for i in ideologies]
    return distances


def gen_candidates(n: int, population: Population) -> List[Candidate]:
    cc = []
    for i in range(n):
        v = population.unit_sample_voter()
        cc.append(Candidate("%s-%d" % (population.party.short_name, i + 1), population.party, v.ideology, 0))
    return cc


def run_election(processes: List[ElectionConstructor]) -> List[ElectionResult]:
    global all_voters, all_candidates
    pop = NDPopulation(np.array([0, 0]), np.array([40, 40]))
    voters = pop.generate_unit_voters(1000)
    candidates = gen_candidates(6, pop)
    candidates.append(Candidate("V", Independents, Ideology(np.random.normal(scale=[1.0, 1.0])), quality=0))

    vv = [v.ideology.distance_from_o() for v in voters]
    all_voters = np.append(all_voters, vv)
    cc = [c.ideology.distance_from_o() for c in candidates]
    all_candidates = np.append(all_candidates, cc)
    ballots = [Ballot(v, candidates, unit_election_config) for v in voters]
    results = [p.run(ballots, set(candidates)) for p in processes]
    return results


if __name__ == '__main__':
    main()
