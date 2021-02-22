from Population import Population
from Voter import Voter
from Ideology import Ideology
from Party import Party
import numpy as np


class NDPopulation(Population):
    def __init__(self, location: np.array, scale: np.array, party: Party):
        super().__init__(party)
        self.location = location
        self.scale = scale
        self.party = party


    def sample_voter(self) -> Voter:
        ideology = np.random.normal(loc=self.location, scale=self.scale)
        return Voter(Ideology(ideology), self.party)
