from Voter import Voter
from Party import Party
from typing import List


class Population:
    def __init__(self, party: Party):
        super().__init__()
        self.party = party

    def sample_voter(self) -> Voter:
        pass

    def generate_voters(self, n: int) -> List[Voter]:
        return list(map(lambda i: self.sample_voter(), range(n)))
