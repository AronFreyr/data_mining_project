import unittest

import gmm


class TestGMM(unittest.TestCase):

    def setUp(self):
        pass

    def test_plot_data(self):
        x = gmm.generate_data_for_gmm(show_plot=False)
        self.assertEqual(500, len(x))


if __name__ == '__main__':
    unittest.main()
