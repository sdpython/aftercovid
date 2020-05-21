"""
Grid of models.
"""
import numpy
from ..data import load_grid_image
from ..data.image_helper import reduce_image


class GridMapSIR:
    """
    Implements a grid of models.

    :param model: epidemic model like :class:`CovidSIR
        <aftercovid.model.CovidSIR>`
    :param name: picture name, must be a binary image
    :param grid_size: new size of the image
    :param gamma: propagation of the epidemics to the neighborhood
    """

    def __init__(self, model, name="france_bin.bmp", grid_size=(10, 12),
                 gamma=0.1):
        self.raw_grid = load_grid_image(name)
        self.grid = reduce_image(self.raw_grid, grid_size)
        self.model = model
        self.gamma = gamma
        self._init()

    def _init(self):
        """
        Initializes the grid.
        """
        self.grid_ = numpy.empty(self.grid.shape).tolist()
        edges = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 0:
                    self.grid_[i][j] = None
                    continue
                self.grid_[i][j] = self.model.copy()

                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        if (i + ii >= 0 and j + jj >= 0 and
                                i + ii < self.grid.shape[0] and
                                j + jj < self.grid.shape[1] and
                                self.grid[ii, jj] == 1):
                            edges.append(((i, j), (i + ii, j + jj)))
        self.edges_ = edges

    def __getitem__(self, name):
        """
        Returns the sum of the metrics all over a grid.

        :param name: quantity name
        :return: float
        """
        g = self.metric(name)
        return g.sum()

    def metric(self, metric="S"):
        """
        Returns one metric for all models on the grid.

        :param name: metric name
        :return: numpy array
        """
        res = numpy.zeros(self.grid.shape, dtype=numpy.float64)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if not self.grid[i, j]:
                    continue
                res[i, j] = self.grid_[i][j][metric]
        return res
