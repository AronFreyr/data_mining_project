import unittest

import gmm
import generate_data
import generate_graphs
from sklearn.datasets import make_moons, make_circles, make_blobs


class TestGMM(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_plots_for_data_example_1(self):
        dimensions = 2
        clusters = 2
        test_gmm = gmm.GMM(clusters)
        x = generate_data.data_example_1(show_plot=False)
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

    def test_get_plots_for_data_example_2(self):
        dimensions = 2
        clusters = 4
        test_gmm = gmm.GMM(clusters)
        x = generate_data.data_example_2(show_plot=False)
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        test_gmm.plot_predictions()
        #generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

    def test_get_plots_for_data_example_3(self):
        dimensions = 2
        clusters = 4
        test_gmm = gmm.GMM(clusters)
        x = generate_data.data_example_3(show_plot=False)
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

    def test_get_plots_for_data_example_5(self):
        dimensions = 2
        clusters = 5
        test_gmm = gmm.GMM(clusters)
        x = generate_data.data_example_5(show_plot=False)
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

    def test_get_plots_for_data_example_6(self):
        dimensions = 2
        clusters = 16
        test_gmm = gmm.GMM(clusters, eps=0.001)
        x = generate_data.data_example_6(show_plot=False)
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

    def test_get_plots_for_data_example_moons(self):
        dimensions = 2
        clusters = 2
        test_gmm = gmm.GMM(clusters)
        #x = generate_data.data_example_9(show_plot=False)
        x = make_moons(n_samples=1000)[0]
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

        circles = make_circles(n_samples=1000, factor=0.5, noise=0.05)

    def test_get_plots_fro_data_example_iris(self):
        dimensions = 2
        clusters = 2
        test_gmm = gmm.GMM(clusters, eps=0.0001)
        # x = generate_data.data_example_9(show_plot=False)
        x = make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')

    def test_get_plots_fro_data_example_7(self):
        dimensions = 3
        clusters = 9
        test_gmm = gmm.GMM(clusters, eps=0.0001)
        x = generate_data.data_example_7(show_plot=False)
        #x = make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]
        test_gmm.fit(x)
        prediction = test_gmm.predict(x)
        test_gmm.make_clusters()
        generate_graphs.plot_predictions(dimensions, x, clusters, prediction, plot_title='Non-robust GMM')


if __name__ == '__main__':
    unittest.main()
