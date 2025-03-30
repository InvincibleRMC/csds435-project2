from typing import override

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.cluster import DBSCAN
from numpy.typing import ArrayLike

class Cluster:

    def __init__(
        self, cluster_count: int
    ) -> None:
        raise NotImplementedError

    def train(self, x_train: ArrayLike, y_train: ArrayLike) -> None:
        raise NotImplementedError

    
    def predict(self, x_test: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class KMeans(Cluster):

    @override
    def __init__(
        self, cluster_count: int,
    ) -> None:
        
        self.cluster = SKLearnKMeans(cluster_count, random_state=0, n_init="auto")
        
    @override
    def train(self, x_train: ArrayLike, y_train: ArrayLike) -> None:
        self.cluster.fit(x_train, y_train)

    @override
    def predict(self, x_test: ArrayLike) -> ArrayLike:
        return self.cluster.predict(x_test)
    

class Spectral(Cluster):

    @override
    def __init__(
        self, cluster_count: int,
    ) -> None:
        
        self.cluster = SpectralClustering(cluster_count, random_state=0)
        
    @override
    def train(self, x_train: ArrayLike, y_train: ArrayLike) -> None:
        self.cluster.fit(x_train, y_train)

    @override
    def predict(self, x_test: ArrayLike) -> ArrayLike:
        return self.cluster.fit_predict(x_test)
    
class DensityBased(Cluster):

    @override
    def __init__(
        self, cluster_count: int,
        isCho: bool = True
    ) -> None:
        
        if isCho:
            self.cluster = DBSCAN(min_samples=4)
        else:
            self.cluster = DBSCAN(eps=0.25, min_samples=4)
        
    @override
    def train(self, x_train: ArrayLike, y_train: ArrayLike) -> None:
        self.cluster.fit(x_train, y_train)

    @override
    def predict(self, x_test: ArrayLike) -> ArrayLike:
        return self.cluster.fit_predict(x_test)