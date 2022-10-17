import unittest

import robust_gmm
import generate_data
import generate_graphs
from sklearn.datasets import make_moons, make_circles, make_blobs


class TestRobustGMM(unittest.TestCase):

    def setUp(self):
        pass

    def test_run_robust_gmm(self):
        x = generate_data.data_example_2(show_plot=False)
        gmm = robust_gmm.RobustGMM(eps=0.001, plot_intermediate_steps=True, plot_step_counter=5)
        gmm.fit(x)
        predictions = gmm.predict(x)
        gmm.make_clusters()
        gmm.plot_predictions()

    def test_get_plots_for_data_example_1(self):
        test_gmm = robust_gmm.RobustGMM()
        x = generate_data.data_example_1(show_plot=False)
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_get_plots_for_data_example_2(self):
        test_gmm = robust_gmm.RobustGMM()
        x = generate_data.data_example_2(show_plot=False)
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_get_plots_for_data_example_3(self):
        test_gmm = robust_gmm.RobustGMM()
        x = generate_data.data_example_3(show_plot=False)
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_get_plots_for_data_example_5(self):
        test_gmm = robust_gmm.RobustGMM()
        x = generate_data.data_example_5(show_plot=False)
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_get_plots_for_data_example_6(self):
        test_gmm = robust_gmm.RobustGMM()
        x = generate_data.data_example_6(show_plot=False)
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_get_plots_for_data_example_moons(self):
        test_gmm = robust_gmm.RobustGMM()
        x = make_moons(n_samples=1000)[0]
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_get_plots_for_data_example_iris(self):
        test_gmm = robust_gmm.RobustGMM(plot_intermediate_steps=True, plot_step_counter=5)
        x = make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]
        test_gmm.fit(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()

    def test_run_robust_gmm_with_3d_data(self):
        x = generate_data.data_example_7(show_plot=False)
        gmm = robust_gmm.RobustGMM()
        gmm.fit(x)
        predictions = gmm.predict(x)
        gmm.make_clusters()
        gmm.plot_predictions()


if __name__ == '__main__':
    unittest.main()
