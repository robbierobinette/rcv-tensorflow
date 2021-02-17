from typing import List
import math
import numpy as np
import numpy.typing as npt


class IdeologyBase:
    def __init__(self, ideology_vec: npt.ArrayLike):
        self.vec = ideology_vec


class Ideology(IdeologyBase):
    def __init__(self, ideology_vec: npt.ArrayLike):
        super().__init__(ideology_vec)

    def length(self) -> float:
        return np.sum(np.abs(self.vec))

    def euclidean_distance(self, rhs: IdeologyBase) -> float:
        deltas = rhs.vec - self.vec
        return np.sqrt(np.sum(deltas * deltas))

    def distance(self, rhs: IdeologyBase) -> float:
        return self.euclidean_distance(rhs)

    # Manhattan distance
    def manhattan_distance(self, rhs: IdeologyBase) -> float:
        l = np.shape(self.vec)[0]
        distance = 0
        for i in range(l):
            distance += abs(self.vec[i] - rhs.vec[i])
        return distance
        # return np.sum(np.abs(self.vec - rhs.vec))
