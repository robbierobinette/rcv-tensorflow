from Party import Party, Independents
from Ideology import Ideology
from Candidate import Candidate
from ElectionConfig import ElectionConfig
import random


class Voter:
    def __init__(self, ideology: Ideology, party: Party):
        self.ideology = ideology
        self.party = party

    def score(self, candidate: Candidate, config: ElectionConfig) -> float:
        score = 200 - self.ideology.distance(candidate.ideology)
        score += candidate.quality

        if self.party == candidate.party:
            score += config.party_loyalty
        elif candidate.party == Independents:
            score += config.independent_bonus

        score += random.normalvariate(0, config.uncertainty)

        if candidate.party == Independents:
            score -= config.wasted_vote_factor

        return score
