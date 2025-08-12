# engine/kdtree_index.py
import numpy as np
from sklearn.neighbors import KDTree
import joblib

class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        if vectors is None or vectors.size == 0:
            raise ValueError("VectorIndex requires a nonempty matrix")
        self.vectors = np.asarray(vectors, dtype=float)
        self.mean_ = self.vectors.mean(axis=0)
        self.std_ = self.vectors.std(axis=0) + 1e-9
        Z = (self.vectors - self.mean_) / self.std_
        self.tree = KDTree(Z, leaf_size=40)

    def query_by_vector(self, v: np.ndarray, k: int = 30):
        v = np.asarray(v, dtype=float)
        Z = (v - self.mean_) / self.std_
        dist, idx = self.tree.query([Z], k=k)
        return dist[0], idx[0]

    def save(self, path: str):
        joblib.dump(
            {"vectors": self.vectors, "mean": self.mean_, "std": self.std_, "tree": self.tree},
            path
        )

    @staticmethod
    def load(path: str):
        obj = joblib.load(path)
        vi = VectorIndex(obj["vectors"])
        vi.mean_ = obj["mean"]
        vi.std_ = obj["std"]
        vi.tree = obj["tree"]
        return vi
