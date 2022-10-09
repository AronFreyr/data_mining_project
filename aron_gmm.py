import numpy as np
from scipy.stats import multivariate_normal

from gmm import generate_data_for_gmm
from generate_graphs import plot_predictions

# WORK IN PROGRESS!!!


def aron_gmm():
    data = generate_data_for_gmm(show_plot=False)

    # Step 1

    n = len(data)
    epsilon = 0.1  # First guess at epsilon
    cluster_nr = 3
    c = cluster_nr

    # mus - random n rows of X
    mus = data[np.random.choice(data.shape[0], c, replace=False)]

    # eye matrices
    covs = np.array([np.eye(data.shape[1]) for _ in range(c)])

    s = 1
    z_hat = np.zeros((n, c))  # Initialize z_hat with zeroes.

    alpha_s = [1/c] * c  # Initial alphas is one divided by nr of clusters.

    z_hat = compute_z_hat(alpha_s, mus, covs, z_hat, data, c)  # Initial computation of z_hat

    while True:

        # Step 2
        for x in range(len(alpha_s)):
            alpha_s[x] = compute_alpha(z_hat[:, x], n)

        for x in range(len(mus)):
            mus[x] = compute_mu(z_hat[x], data, c)

        # Step 3
        for x in range(len(covs)):
            covs[x] = compute_covariance(z_hat[x], mus[x], data, c)

        # Step 3+
        new_z_hat = compute_z_hat(alpha_s, mus, covs, z_hat, data, c)

        # Step 4
        comparison = np.linalg.norm([z_hat - new_z_hat])
        z_hat = new_z_hat.copy()
        if comparison < epsilon:
            break

    prediction = predict(data, mus, covs)
    #print(prediction)

    plot_predictions(data.shape[1], data, z_hat, c, prediction)


def compute_z_hat(alphas, mus, covs, z_hat, data, c):
    calc_z_hat = z_hat.copy()
    for x in range(c):
        pdf = multivariate_normal.pdf(data, mean=mus[x], cov=covs[x])  # pdf for every data point
        calc_z_hat[:, x] = alphas[x] * pdf
    calc_z_hat = calc_z_hat / np.sum(calc_z_hat, axis=1, keepdims=True)
    return calc_z_hat


def compute_alpha(z_hat_k, n):
    # Equation 6
    alpha_k = np.sum(z_hat_k) / n
    return alpha_k


def compute_mu(z_hat_k, data, c):
    # Equation 8
    the_sum = 0
    for i in range(c):
        the_sum += z_hat_k[i] * data[i]
    mu_k = the_sum / np.sum(z_hat_k)
    return mu_k


def compute_covariance(z_hat_k, mu_k, data, c):
    # Equation 9
    vector = data - mu_k
    the_sum = 0
    for i in range(c):
        the_sum += np.dot(z_hat_k[i] * vector.T, vector)
    cov_k = the_sum / np.sum(z_hat_k)
    return cov_k


def predict(Y, mus, covs):
    # predict on unseen data
    # find to which cluster Mahalenobis distance is minimized
    dists = []
    for p in range(Y.shape[0]):
        lista = []
        for i in range(len(mus)):
            delta = Y[p] - mus[i]
            m = np.dot(np.dot(delta, np.linalg.inv(covs[i])), delta)
            lista.append(np.sqrt(m))
        dists.append(np.argmin(lista))
    return np.array(dists)


if __name__ == '__main__':
    aron_gmm()
