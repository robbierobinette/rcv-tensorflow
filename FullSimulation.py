import matplotlib.pyplot as plt

from Ballot import Ballot
from DefaultConfigOptions import *
from PartyPrimaryElection import PartyPrimaryElection
from ElectionResult import ElectionResult
from DistrictData import DistrictVotingRecord, DistrictData
from InstantRunoffElection import InstantRunoffResult, InstantRunoffElection


def main():
    dd = DistrictData("data-5vPn3.csv")
    dec = default_election_config

    winners: List[List[ElectionResult]] = []
    for i in range(11):
        dvr_list: List[DistrictVotingRecord] = list(dd.dvr.values())
        print("iteration %d" % i)
        c = 0
        for dvr in dvr_list:
            pop = dvr.population(dec.partisanship, dec.skew_factor, dec.stddev)
            winners.append(run_election(pop))
            c += 1
            print(".", end="")
            if c % 100 == 0:
                print("")
        print("")

    processes = ["Party Primary", "Instant Runoff"]
    for process_index in range(2):
        rep_ideology = get_plot_column(winners, process_index, Republicans)
        dem_ideology = get_plot_column(winners, process_index, Democrats)

        plt.hist([dem_ideology, rep_ideology],
                 stacked=True,
                 density=True,
                 bins=30,
                 color=["blue", "red"],
                 label=["Democrats", "Republicans"],
                 )
        plt.title(processes[process_index])
        plt.xlabel('ideology')
        plt.ylabel('count')
        plt.show()


def get_plot_column(winners: List[List[ElectionResult]], process_index: int, party: Party) -> List[float]:
    return [r[process_index].winner().ideology.vec[0] for r in winners if r[process_index].winner().party == party]


def gen_candidates(population: PopulationGroup) -> List[Candidate]:
    cc = []
    for i in range(0, 3):
        v = population.sample_voter()
        cc.append(Candidate("%s-%d" % (population.party.short_name, i + 1), population.party, v.ideology, 0))
    return cc


def run_election(pop: CombinedPopulation) -> List[ElectionResult]:
    voters = pop.generate_voters(1000)
    candidates = set(gen_candidates(pop.republicans) + gen_candidates(pop.democrats))
    ballots = [Ballot(v, candidates, default_election_config) for v in voters]
    plurality = PartyPrimaryElection(ballots, candidates, pop, default_election_config)
    irv = InstantRunoffElection(ballots, candidates)
    return [plurality.result(), irv.result()]


if __name__ == '__main__':
    main()
