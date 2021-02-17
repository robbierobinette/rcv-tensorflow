from Election import ElectionResult, Election, BallotIter, Ballot
from typing import List, Iterable, Union, Set
from Candidate import Candidate
from PluralityElection import PluralityElection, PluralityResult


class InstantRunoffResult(ElectionResult):
    def __init__(self, ordered_candidates: List[Candidate], rounds: List[PluralityResult]):
        super().__init__(ordered_candidates)
        self.rounds = rounds


class InstantRunoffElection(Election):
    def __init__(self, ballots: BallotIter, candidates: Set[Candidate]):
        super().__init__(ballots, candidates)

    def result(self) -> InstantRunoffResult:
        return self.compute_result()

    def compute_result(self) -> InstantRunoffResult:
        active_candidates = self.candidates
        rounds = []
        losers = []
        while len(active_candidates) > 1:
            plurality = PluralityElection(self.ballots, self.candidates)
            r = plurality.result()
            rounds.append(r)
            losers.append(r.ordered_candidates[-1])
            active_candidates.remove(r.ordered_candidates[-1])

        winners = list(losers.__reversed__())

        return InstantRunoffResult(winners, rounds)



