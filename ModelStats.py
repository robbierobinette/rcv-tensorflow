from typing import List
from Candidate import Candidate


class ModelStats:
    def __init__(self):
        self.model_count = 1e-5
        self.model_winners = 1e-5
        self.random_count = 1e-5
        self.random_winners = 1e-5
        self.model_winner_distance = 0
        self.random_winner_distance = 0
        self.winners: List[Candidate] = []
        self.candidates: List[List[Candidate]] = []
        self.balance: List[float] = []

    def reset(self):
        self.model_count = 1e-5
        self.model_winners = 1e-5
        self.random_count = 1e-5
        self.random_winners = 1e-5
        self.model_winner_distance = 0
        self.random_winner_distance = 0
        self.winners = []
        self.candidates = []
        self.balance = []

    def update(self, winner: Candidate, candidates: List[Candidate], balance: float = 0):
        self.winners.append(winner)
        self.candidates.append(candidates)
        self.balance.append(balance)

        for c in candidates:
            if c.name[0] == 'm':
                self.add_model()
            else:
                self.add_random()

        if winner.name[0] == 'm':
            self.add_model_winner(winner)
        else:
           self.add_random_winner(winner)

    def add_random(self):
        self.random_count += 1

    def add_model(self):
        self.model_count += 1

    def add_random_winner(self, w: Candidate):
        self.random_winners += 1
        self.random_winner_distance += w.ideology.distance_from_o()

    def add_model_winner(self, w: Candidate):
        self.model_winners += 1
        self.model_winner_distance += w.ideology.distance_from_o()

    def print(self, label: str, global_step: int):
        print("%15s %6d, %5d " %
              (label,
               global_step,
               len(self.winners)), end="")

        print("random %6d/%6d %5.2f%% O: %5.2f" %
              (self.random_count,
               self.random_winners,
               100 * self.random_winners / self.random_count,
               self.random_winner_distance / self.random_winners), end='')

        print(" model %6d/%6d %5.2f%% O: %5.2f" %
              (self.model_count,
               self.model_winners,
               100 * self.model_winners / self.model_count,
               self.model_winner_distance / self.model_winners), end='')

        print(" chance of model_winner = %5.2f%%" % (
                100 * self.model_winners / (self.model_winners + self.random_winners)),
              flush=True)
