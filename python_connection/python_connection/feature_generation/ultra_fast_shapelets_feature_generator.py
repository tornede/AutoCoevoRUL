from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


def rolling_window(x, window):
    """
    Function to roll an array given a specified window. As result, all possible subsequences of length window are
    returned in the resulting matrix

    :param x: Instance features
    :param window: The window size
    :return: Returns the rolled out array
    """
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


"""
Implementation of the ultra fast shapelets algorithm for univariate timeseries from Wistuba, M., Grabocka, J., & Schmidt-Thieme, L. (2015). Ultra-fast shapelets for time series classification. arXiv preprint arXiv:1503.05018. 
The data is expected to come in a numpy array with one row per instance and one column per timestep. All series are expected to have the same length and a value for each timestep.
"""
class UltraFastShapeletsFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, keep_candidates_percentage = 0.1, random_seed=None):
        assert 0 < keep_candidates_percentage <= 1, "The percentage of kept candidates must be greater in (0,1]."

        self.keep_candidates_percentage = keep_candidates_percentage
        self.random_state = random_seed

    def fit(self, X, y):
        # Get transformed dataset
        self.shapelets = self._extract_shapelets(X)
        return self

    def transform(self, X):
        X_transf = self._shapelet_transform(X)
        return X_transf

    def _extract_shapelets(self, X):
        # Get number of subsequences
        n = X.shape[0]
        m = X.shape[1]

        # Use minimal filter size to assure comparability to LF, GF and LS
        filter_size = int(0.1 * m)
        #self.p = int(3 * np.log10(n * (m - filter_size + 1) * (self._num_classes - 1)))
        self.p = int(n * self.keep_candidates_percentage)

        # Clip p to 1 (lower bound)
        if self.p < 1:
            self.p = 1

        if len(X.shape) == 3:
            dimensions = X.shape[2]
        else:
            dimensions = 1

        c = np.zeros([m - 2])
        for i in range(len(c)):
            l = i + 3
            c[i] = self._get_number_of_candidates(n, m, l)

        # Calculate ratio f
        f = self.p * (1 / np.sum(c))

        # Set random state if one is provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Variable used to ensure that round(f * cl) for each l sums up to p
        ps = np.zeros_like(c)
        for i in range(len(ps)):
            ps[i] = int(round(f * c[i]))
        if np.sum(ps) < self.p:
            # Distribute the missing elements to ps randomly
            missing_elements = int(self.p - np.sum(ps))
            for _ in range(missing_elements):
                ps[np.random.randint(len(ps))] += 1
        elif np.sum(ps) > self.p:
            # Collect the surplus elements from ps randomly
            surplus_elements = int(np.sum(ps) - self.p)
            for _ in range(surplus_elements):
                # Determine the start index for round robin extraction search randomly
                start_index = np.random.randint(len(ps))
                for i in range(len(ps)):
                    curr_index = (start_index + i) % len(ps)
                    if ps[curr_index] > 0:
                        ps[curr_index] -= 1
                        break
        ps = ps.astype('int')

        # Get shapelets
        subsequences = []
        for l in range(2, m):
            for k in range(dimensions):
                for _ in range(ps[l - 2]):
                    i = np.random.randint(n)
                    j = np.random.randint(m - l)

                    # Extract shapelets from data
                    if dimensions == 1:
                        shapelet = X[i, j:j + l]
                        subsequences.append(shapelet)
                    else:
                        shapelet = X[i, j: j + l, k]
                        subsequences.append((shapelet, k))
        return subsequences

    @staticmethod
    def _min_norm_distance(x, s):
        # This function is used within UFS for the univariate case
        l = len(s)

        mu = np.mean(x)
        sigma = np.std(x)
        epsilon = 0.000001 #to avoid division by zero
        # for i in range(m - l + 1):
        return np.min(np.linalg.norm(s - ((rolling_window(x, l) - mu) / (sigma + epsilon)), axis=-1))

    @staticmethod
    def _min_distance(x, s):
        # This function is used within UFS for the multivariate case
        l = len(s)
        return np.min(np.linalg.norm(s - rolling_window(x, l), axis=-1))

    def _shapelet_transform(self, X):
        # Generate transformed dataset
        n = X.shape[0]
        is_multivariate = len(X.shape) == 3

        output_x = np.zeros([n, self.p])
        for j in range(self.p):
            subseq = self.shapelets[j]
            for i in range(n):
                if is_multivariate:
                    output_x[i, j] = self._min_distance(X[i, subseq[1]], subseq[0])
                else:
                    output_x[i, j] = self._min_norm_distance(X[i], subseq)
        return output_x

    @staticmethod
    def _get_number_of_candidates(n, m, l):
        assert l > 0, "The window length has to be greater than zero."

        if l > m:
            l = m
        return n * (m - l + 1)