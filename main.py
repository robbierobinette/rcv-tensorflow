from PopulationGroup import PopulationGroup
from Party import *
from Voter import Voter
from CombinedPopulation import CombinedPopulation
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def main():
    population_groups = [
        PopulationGroup(Republicans, 30, 30, .4),
        PopulationGroup(Independents, 0, 30, .2),
        PopulationGroup(Democrats, -30, 30, .4)
    ]

    combined_population = CombinedPopulation(population_groups)
    voters = list(map(lambda i: combined_population.sample_voter(), range(1000)))

    iv = filter(lambda v: v.party == Independents, voters)
    ii = list(map(lambda v: v.ideology.vec[0], iv))

    dv = filter(lambda v: v.party == Democrats, voters)
    di = list(map(lambda v: v.ideology.vec[0], dv))

    rv = filter(lambda v: v.party == Republicans, voters)
    ri = list(map(lambda v: v.ideology.vec[0], rv))

    plt.hist([di, ii, ri],
             stacked=True,
             density=True,
             bins=30,
             color=["red", "gray", "blue"],
             label=["Democrats", "Independents", "Republicans"],
             )
    plt.xlabel('ideology')
    plt.ylabel('count')
    plt.show()


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
