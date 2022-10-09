import unittest

import robust_gmm
import generate_data
import generate_graphs


class TestRobustGMM(unittest.TestCase):

    def setUp(self):
        pass

    def test_run_robust_gmm(self):
        x = generate_data.data_example_5(show_plot=False)
        gmm = robust_gmm.RobustGMM(eps=0.001, plot_intermediate_steps=True, plot_step_counter=5)
        gmm.fit(x)
        predictions = gmm.predict(x)
        # print(predictions)
        gmm.make_clusters()
        gmm.plot_predictions()
        # generate_graphs.plot_predictions(gmm.d, x, gmm.c, gmm.clusters)

    def test_run_robust_gmm_with_3d_data(self):
        x = generate_data.data_example_1_3D(show_plot=False)
        gmm = robust_gmm.RobustGMM()
        gmm.fit(x)
        predictions = gmm.predict(x)
        # print(predictions)
        gmm.make_clusters()
        gmm.plot_predictions()
        #generate_graphs.plot_predictions(gmm.d, x, gmm.c, gmm.clusters)


if __name__ == '__main__':
    unittest.main()
