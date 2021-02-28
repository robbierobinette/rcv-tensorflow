from ElectionConstructor import ElectionConstructor
from ModelStats import ModelStats
import pickle

class ProcessResult:
    def __init__(self, process: ElectionConstructor, bins: int, dim: int, stats: ModelStats):
        self.process = process
        self.dim = dim
        self.bins = bins
        self.stats = stats
        self.label = "%15s ib%3d %dD" % (process.name, bins, dim)

    def print(self):
        self.stats.print(self.label, 0)

    def name(self) -> str:
        return "%s-%03d-%dD" % (self.process.name, self.bins, self.dim)

    def save(self, dir: str):
        with open("%s/%s.p" % (dir, self.name()), "wb") as f:
            pickle.dump(self, f)
