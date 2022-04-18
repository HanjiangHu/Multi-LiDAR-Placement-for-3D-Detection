import numpy as np
import scipy



class PseudoLIDAR:
    def __init__(self, map, distance=5):
        self.dim = 2
        self.lb = np.zeros(2)
        self.ub = np.max(map.shape) * np.ones(2)
        self.lb -= 1
        self.ub -= 1
        self.map = map
        self.distance = distance

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        return -self.compute_entropy_points_in_distance(x)

    def compute_entropy_points_in_distance(self, placement):
        shape = self.map.shape
        xy = np.mgrid[0 : shape[0] : 1, 0 : shape[1] : 1].reshape(2, -1).T
        distances = scipy.spatial.distance.cdist(placement.reshape(-1, 2), xy)
        indices = distances < self.distance
        points = xy[indices.flatten(), :].T.tolist()
        probs = self.map[tuple(points)]
        entropy = np.sum(scipy.special.entr(probs))
        return entropy


if __name__ == "__main__":
    print("test lidar")
