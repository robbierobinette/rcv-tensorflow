from typing import List, Set
from CandidateScore import CandidateScore
from Candidate import Candidate
from Voter import Voter
from ElectionConfig import ElectionConfig


class Ballot:
    def __init__(self, voter: Voter, candidates: List[Candidate], config: ElectionConfig):
        self.voter = voter

        scores = list(map(lambda c: voter.score(c, config), candidates))
        cs = list(map(lambda c: CandidateScore(c[0], c[1]), zip(candidates, scores)))
        cs.sort(key=lambda c: c.score, reverse=True)
        self.ordered_candidates = cs

    def active_winner(self, active_candidates: Set[Candidate]) -> Candidate:
        fc = filter(lambda c: c.candidate in active_candidates, self.ordered_candidates)
        return next(fc).candidate

    def winner(self) -> Candidate:
        return self.ordered_candidates[0].candidate

    def print(self):
        for cs in self.ordered_candidates:
            print("\t %6s ideology: % 7.2f score: % 7.2f" % (cs.candidate.name, cs.candidate.ideology.vec[0], cs.score))

