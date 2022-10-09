import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_predictions(dimensions, data, c_input, clusters, plot_title='Clustering results'):
    """
    Creates a plot of the resulting data after it has gone through GMM.
    Can create 1D, 2D and 3D plots.
    :param dimensions: The dimensionality of the plot. Can only be 1, 2 or 3.
    :param data: The data points that will be plotted on the graph.
    :param c_input: Something to do with cluster calculations.
    :param clusters: A list of the clusters that the data points belong to according to GMM.
    :param plot_title: The title of the plot. Default is 'Clustering results'.
    :return: Nothing
    """

    def _create_ellipses(x_coords, y_coords, ellipse_color='b'):
        """
        Function for creating ellipses around clusters.
        :param x_coords: All x coordinates of our data as a list.
        :param y_coords: All y coordinates of our data as a list.
        :ellipse_color: color of the ellipses. Default is blue.
        :return: A drawn ellipse object.
        """

        x_length = x_coords.max() - x_coords.min()
        y_length = y_coords.max() - y_coords.min()
        mid_point_x = (x_coords.max() + x_coords.min()) / 2
        mid_point_y = (y_coords.max() + y_coords.min()) / 2
        cluster_ellipse = Ellipse(xy=(mid_point_x, mid_point_y), width=x_length, height=y_length, fc='None',
                                  edgecolor=ellipse_color)

        # Makes sure that the ellipses are always on top, otherwise they can be hard to see.
        cluster_ellipse.set_zorder(10)
        return cluster_ellipse

    np.random.seed(1)

    if dimensions > 3 or dimensions <= 0:
        raise ValueError('This method can only plot 1-3D data')

    if dimensions == 1:
        X_with_preds = np.c_[data, clusters]
        colors = [np.random.rand(3, ) for _ in range(c_input)]
        for i in range(data.shape[0]):
            plt.plot(data[i], '.', alpha=1, color=colors[int(X_with_preds[i][1])])
        plt.grid()
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    elif dimensions == 2:
        fig, ax = plt.subplots()
        unique_clusters = list(set(clusters))
        for i in unique_clusters:
            c = data[clusters == i]
            x_coordinates = c[:, 0]
            y_coordinates = c[:, 1]

            label = i + 1 if len(unique_clusters) < 6 else None
            ax.scatter(x_coordinates, y_coordinates, color=np.append(np.random.rand(3,), 0.5),
                       edgecolor='k', linewidth=0.3, label=label, zorder=2)

            ellipse = _create_ellipses(x_coordinates, y_coordinates)
            ax.add_artist(ellipse)
        if len(unique_clusters) < 6:
            ax.legend()
        plt.grid()

    elif dimensions == 3:
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_clusters = list(set(clusters))
        for i in unique_clusters:
            c = data[clusters == i]
            label = i + 1 if len(unique_clusters) < 10 else None
            ax.scatter(c[:, 0], c[:, 1], c[:, 2], color=np.append(np.random.rand(3,), 0.5),
                       edgecolor='black', label=label, zorder=2)
        plt.grid()
        if len(unique_clusters) < 10:
            ax.legend()

    plt.title(plot_title)
    plt.show()
