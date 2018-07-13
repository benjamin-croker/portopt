import unittest
import numpy as np
import portopt

# tests adapted from examples in
# https://github.com/lequant40/portfolio_allocation_js


class TestSharpeOptim(unittest.TestCase):
    def setUp(self):
        self.cov = np.array(
            [[0.010, 0.010, -0.0018, 0.0024, 0.0016, 0.0048],
             [0.010, 0.0625, 0.0135, 0.009, 0.002, 0.008],
             [-0.0018, 0.0135, 0.0324, 0.00432, -0.00288, 0.00864],
             [0.0024, 0.009, 0.00432, 0.0144, 0.0096, 0.00192],
             [0.0016, 0.002, -0.00288, 0.0096, 0.0064, 0.00256],
             [0.0048, 0.008, 0.00864, 0.00192, 0.00256, 0.0256]]
        )
        self.rets = np.array([0.15, 0.18, 0.20, 0.11, 0.13, 0.12])

    def test_optimise_sharpe(self):
        expected_weights = np.array([0.3318, 0, 0.2443, 0, 0.4239, 0])
        allocs = portopt.effecient_frontier(self.rets, self.cov, 0.08, n=200)
        best = portopt.optim_sharpe(allocs, self.rets, self.cov, 0.08, n=200)
        np.testing.assert_almost_equal(expected_weights, best['w'], decimal=2)


if __name__ == '__main__':
    unittest.main()
