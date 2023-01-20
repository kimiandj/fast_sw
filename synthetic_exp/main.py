import numpy as np
import os
import torch
import itertools
import pickle5 as pickle
from time import time
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
matplotlib.rcParams.update({'font.size': 16})

np.random.seed(10)


def montecarlo_sw(X, Y, L=100, p=2):
    """
    Computes the Monte Carlo estimation of Sliced-Wasserstein distance between empirical distributions
    """
    N, d = X.shape
    order = p
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    # Project data
    theta = torch.randn(L, d)
    theta = theta / torch.linalg.norm(theta, dim=1)[:, None]  # normalization (theta is in the unit sphere)
    theta = torch.t(theta)
    xproj = torch.matmul(X, theta)
    yproj = torch.matmul(Y, theta)
    # Sort projected data
    xqf, _ = torch.sort(xproj, dim=0)
    yqf, _ = torch.sort(yproj, dim=0)
    # Compute expected SW distance
    sw_dist = torch.mean(torch.abs(xqf - yqf) ** order)
    sw_dist = sw_dist ** (1/order)
    return sw_dist


def approximate_sw(X, Y, centering=True):
    """
    Approximates SW with Wasserstein distance between Gaussian approximations
    """
    d = X.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    if centering:
        # Center the data
        mean_X = torch.mean(X, dim=0)
        mean_Y = torch.mean(Y, dim=0)
        X = X - mean_X
        Y = Y - mean_Y
    # Approximate SW
    m2_Xc = torch.mean(torch.linalg.norm(X, dim=1) ** 2) / d
    m2_Yc = torch.mean(torch.linalg.norm(Y, dim=1) ** 2) / d
    sw = torch.abs(m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2))
    return sw


def main_indep(distribution, dims, n_samples, n_runs):
    n_dims = len(dims)
    errors_sw = np.zeros((n_runs, n_dims))
    errors_sw_c = np.zeros((n_runs, n_dims))
    for idim in range(n_dims):
        d = dims[idim]
        if distribution == "gaussian":
            # Set parameters of Gaussian distributions
            mean_X = 1 + np.random.randn(d)
            covfactor_X = 1
            mean_Y = 1 + np.random.randn(d)
            covfactor_Y = 10

            # Compute analytical SW between the Gaussian distributions
            true_sw = (1 / d * np.linalg.norm(mean_X - mean_Y) ** 2 + (
                        covfactor_X ** (1 / 2) - covfactor_Y ** (1 / 2)) ** 2) ** (1 / 2)
            # Compute analytical SW between the centered Gaussian distributions
            true_sw_c = np.abs(covfactor_X ** (1 / 2) - covfactor_Y ** (1 / 2))

            # Compute approximation errors for different runs
            for irun in range(n_runs):
                print("Run {} for dimension {}".format(irun + 1, d))
                # Generate the two datasets from independent Gaussian distributions
                X = np.sqrt(covfactor_X) * np.random.normal(size=(n_samples, d)) + mean_X
                Y = np.sqrt(covfactor_Y) * np.random.normal(size=(n_samples, d)) + mean_Y

                # Approximate SW with our methodology
                approx_sw = approximate_sw(X, Y, centering=False)  # we do not center the data
                approx_sw_c = approximate_sw(X, Y, centering=True)  # we center the data

                # Store approximation errors
                errors_sw[irun, idim] = np.abs(approx_sw - true_sw)
                errors_sw_c[irun, idim] = np.abs(approx_sw_c - true_sw_c)

                # Save approximation errors
                with open("gaussian_centering=0", "wb") as f:
                    pickle.dump(errors_sw, f, pickle.HIGHEST_PROTOCOL)
                with open("gaussian_centering=1", "wb") as f:
                    pickle.dump(errors_sw_c, f, pickle.HIGHEST_PROTOCOL)
        elif distribution == "gamma":
            # Set parameters of Gamma distributions
            shapes_X = np.random.uniform(low=1, high=5, size=d)
            shapes_Y = np.random.uniform(low=5, high=10, size=d)
            scale_X = 2.0
            scale_Y = 3.0

            # Compute approximation errors for different runs
            for irun in range(n_runs):
                print("Run {} for dimension {}".format(irun + 1, d))
                # Generate the two datasets from independent Gamma distributions
                X = np.zeros((n_samples, d))
                Y = np.zeros((n_samples, d))
                for i in range(d):
                    X[:, i] = np.random.gamma(shape=shapes_X[i], scale=scale_X, size=n_samples)
                    Y[:, i] = np.random.gamma(shape=shapes_Y[i], scale=scale_Y, size=n_samples)

                # Approximate the true SW with Monte Carlo based on 2*10^4 projections
                true_sw = montecarlo_sw(X, Y, L=20000)  # we do not center the data
                mean_X = np.mean(X, axis=0)
                mean_Y = np.mean(Y, axis=0)
                true_sw_c = montecarlo_sw(X - mean_X, Y - mean_Y, L=20000)  # we center the data

                # Approximate SW with our methodology
                approx_sw = approximate_sw(X, Y, centering=False)  # we do not center the data
                approx_sw_c = approximate_sw(X, Y, centering=True)  # we center the data

                # Store approximation errors
                errors_sw[irun, idim] = np.abs(approx_sw - true_sw)
                errors_sw_c[irun, idim] = np.abs(approx_sw_c - true_sw_c)

                # Save approximation errors
                with open("gam_centering=0", "wb") as f:
                    pickle.dump(errors_sw, f, pickle.HIGHEST_PROTOCOL)
                with open("gam_centering=1", "wb") as f:
                    pickle.dump(errors_sw_c, f, pickle.HIGHEST_PROTOCOL)
    return errors_sw, errors_sw_c


def run_indep():
    dims = np.array([10, 50, 100, 500, 1000])
    n_samples = 10000
    n_runs = 100

    errors_gauss, errors_gauss_c = main_indep(distribution="gaussian", dims=dims,
                                              n_samples=n_samples, n_runs=n_runs)  # Gaussian setting
    errors_gam, errors_gam_c = main_indep(distribution="gamma", dims=dims,
                                          n_samples=n_samples, n_runs=n_runs)  # Gamma setting

    # Plot approximation errors vs. dimension for noncentered data
    plt.figure(figsize=[5, 4])
    plt.plot(dims, errors_gauss.mean(axis=0), '-o', label="Gaussian", lw=2, ms=10)
    plt.fill_between(dims, np.percentile(errors_gauss, 10, axis=0), np.percentile(errors_gauss, 90, axis=0), alpha=0.2)
    plt.plot(dims, errors_gam.mean(axis=0), '-x', label="Gamma", lw=2, ms=10)
    plt.fill_between(dims, np.percentile(errors_gam, 10, axis=0), np.percentile(errors_gam, 90, axis=0), alpha=0.2)
    plt.xlabel("dimension")
    plt.ylabel("approximation error")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figures/notcentered_nsamples={}_nruns={}.pdf'.format(n_samples, n_runs))

    # Plot approximation errors vs. dimension for centered data
    plt.figure(figsize=[5, 4])
    plt.plot(dims, errors_gauss_c.mean(axis=0), '-o', label="Gaussian", lw=2, ms=10)
    plt.fill_between(dims, np.percentile(errors_gauss_c, 10, axis=0), np.percentile(errors_gauss_c, 90, axis=0), alpha=0.2)
    plt.plot(dims, errors_gam_c.mean(axis=0), '-x', label="Gamma", lw=2, ms=10)
    plt.fill_between(dims, np.percentile(errors_gam_c, 10, axis=0), np.percentile(errors_gam_c, 90, axis=0), alpha=0.2)
    plt.xlabel("dimension")
    plt.ylabel("approximation error")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figures/centered_nsamples={}_nruns={}.pdf'.format(n_samples, n_runs))

    # Compute slopes
    print("We approximate the slopes with linear regression.")
    log_errors = np.log(errors_gauss_c)
    print("Gaussian slopes: {}".format(linregress(np.log(dims), log_errors.mean(axis=0))))
    log_errors = np.log(errors_gam_c)
    print("Gamma slopes: {}".format(linregress(np.log(dims), log_errors.mean(axis=0))))


def main_ar(distribution, dims, alphas, n_samples, n_runs):
    burnin = 10000
    n_dims = len(dims)
    n_alphas = len(alphas)
    errors_approx_sw = np.zeros((n_runs, n_dims, n_alphas))

    # Compute approximation errors over several runs for different dimension and alpha values
    for irun in range(n_runs):
        for idim in range(n_dims):
            d = dims[idim]
            for ialpha in range(n_alphas):
                alpha = alphas[ialpha]
                print("Run {} for dimension {} and alpha {}".format(irun + 1, d, alpha))
                if distribution == "gaussian":
                    var1 = 1
                    Z1 = np.random.normal(0, np.sqrt(var1), size=(n_samples, burnin + d))
                    Z2 = np.random.normal(0, np.sqrt(var1), size=(n_samples, burnin + d))
                elif distribution == "student":
                    df1 = 10
                    Z1 = np.random.standard_t(df=df1, size=(n_samples, burnin + d))
                    Z2 = np.random.standard_t(df=df1, size=(n_samples, burnin + d))
                # Generate two datasets from the same autoregressive model
                X = np.zeros((n_samples, d))
                Y = np.zeros((n_samples, d))
                X_current = Z1[:, 0]
                Y_current = Z2[:, 0]
                print("Started generating data.")
                for i in range(burnin + d - 1):
                    X_current = alpha * X_current + Z1[:, i]
                    Y_current = alpha * Y_current + Z2[:, i]
                    # We discard the first 'burnin' steps
                    if i >= burnin:
                        X[:, i - burnin] = X_current
                        Y[:, i - burnin] = Y_current
                print("Finished generating data.")

                # Approximate SW with our methodology
                approx_sw = approximate_sw(X, Y, centering=True)
                mean_X = np.mean(X, axis=0)
                mean_Y = np.mean(Y, axis=0)
                approx_sw += np.linalg.norm(mean_X - mean_Y) ** 2 / d

                # Exact SW is zero since the two datasets are generated from the same AR(1) model
                true_sw = 0

                # Compute approximation error
                errors_approx_sw[irun, idim, ialpha] = np.abs(approx_sw - true_sw)
                # Save approximation error
                with open(distribution + "_weakdep", "wb") as f:
                    pickle.dump(errors_approx_sw, f, pickle.HIGHEST_PROTOCOL)
    return errors_approx_sw


def run_ar():
    dims = np.array([10, 50, 100, 500, 1000])
    distributions = ["gaussian", "student"]
    alphas = np.linspace(0, 1, 5)
    alphas = alphas[1:-1]  # we remove the special cases alpha = 0 and alpha = 1
    n_samples = 10000
    n_runs = 100

    for distribution in distributions:
        errors_ar = main_ar(distribution=distribution, alphas=alphas, dims=dims, n_samples=n_samples, n_runs=n_runs)
        # Compute slopes
        print("We approximate the slopes with linear regression.")
        log_errors_ar = np.log(errors_ar)
        for k in range(len(alphas)):
            print("Slope for alpha {}: {}".format(alphas[k], linregress(np.log(dims), log_errors_ar[:, :, k].mean(axis=0))))

        # Plot average approximation errors vs. dimension
        marker = itertools.cycle(('-x', '-o', '-^'))
        plt.figure(figsize=[5, 4])
        for k in range(len(alphas)):
            plt.plot(dims, errors_ar[:, :, k].mean(axis=0), next(marker), label=r"$\alpha = {}$".format(alphas[k]), lw=2, ms=10)
            plt.fill_between(dims, np.percentile(errors_ar[:, :, k], 10, axis=0), np.percentile(errors_ar[:, :, k], 90, axis=0), alpha=0.2)
        plt.xlabel("dimension")
        plt.ylabel("approximation error")
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "{}_ar_nsamples={}_nruns={}.pdf".format(distribution, n_samples, n_runs)))


def main_comparison(dims, projs, n_samples, n_runs):
    n_dims = len(dims)
    n_projs = len(projs)

    errors_approx_sw = np.zeros((n_runs, n_dims))
    errors_mc_sw = np.zeros((n_runs, n_dims, n_projs))
    time_approx_sw = np.zeros((n_runs, n_dims))
    time_mc_sw = np.zeros((n_runs, n_dims, n_projs))

    for idim in range(n_dims):
        d = dims[idim]
        # Set parameters of Gamma distributions
        shapes_X = np.random.uniform(low=1, high=5, size=d)
        shapes_Y = np.random.uniform(low=5, high=10, size=d)
        scale_X = 2.0
        scale_Y = 3.0

        for irun in range(n_runs):
            print("Run {} for dimension {}".format(irun + 1, d))
            # Generate the two datasets from independent Gamma distributions
            X = np.zeros((n_samples, d))
            Y = np.zeros((n_samples, d))
            for i in range(d):
                X[:, i] = np.random.gamma(shape=shapes_X[i], scale=scale_X, size=n_samples)
                Y[:, i] = np.random.gamma(shape=shapes_Y[i], scale=scale_Y, size=n_samples)

            # Approximate the true SW with Monte Carlo based on 2*10^4 projections
            true_sw = montecarlo_sw(X, Y, L=20000)

            # Approximate SW with Monte Carlo approximation based on different number of projections
            for iproj in range(n_projs):
                start = time()
                mc_sw = montecarlo_sw(X, Y, L=projs[iproj])
                # Store computation time for Monte Carlo
                time_mc_sw[irun, idim, iproj] = time() - start
                # Compute approximation error
                errors_mc_sw[irun, idim, iproj] = np.abs(mc_sw - true_sw)

            # Approximate SW with our methodology
            start = time()
            mean_X = np.mean(X, axis=0)
            mean_Y = np.mean(Y, axis=0)
            mean_term = np.linalg.norm(mean_X - mean_Y) ** 2 / d
            m2_Xc = np.mean(np.linalg.norm(X - mean_X, axis=1) ** 2) / d
            m2_Yc = np.mean(np.linalg.norm(Y - mean_Y, axis=1) ** 2) / d
            approx_sw = (mean_term + (m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2)) ** 2) ** (1/2)
            # Store computation time for our methodology
            time_approx_sw[irun, idim] = time() - start
            # Compute approximation error
            errors_approx_sw[irun, idim] = np.abs(approx_sw - true_sw)

            # Save errors
            with open("time_errapprox", "wb") as f:
                pickle.dump(errors_approx_sw, f, pickle.HIGHEST_PROTOCOL)
            with open("time_errmc", "wb") as f:
                pickle.dump(errors_mc_sw, f, pickle.HIGHEST_PROTOCOL)
            # Save computation time
            with open("time_approx", "wb") as f:
                pickle.dump(time_approx_sw, f, pickle.HIGHEST_PROTOCOL)
            with open("time_mc", "wb") as f:
                pickle.dump(time_mc_sw, f, pickle.HIGHEST_PROTOCOL)
    return errors_approx_sw, errors_mc_sw, time_approx_sw, time_mc_sw


def run_comparison():
    dims = np.array([10, 50, 100, 500, 1000])
    projs = np.array([100, 1000, 5000])
    n_runs = 100
    n_samples = 10000

    errors_approx, errors_mc, time_approx, time_mc = main_comparison(dims=dims, projs=projs,
                                                                     n_samples=n_samples, n_runs=n_runs)

    # Plot average approximation errors vs. dimension
    plt.figure(figsize=[6, 4])
    marker = itertools.cycle(('-x', '-o', '-^', '-*', '--.'))
    for l in range(len(projs)):
        plt.plot(dims, errors_mc[:, :, l].mean(axis=0), next(marker), label="Monte Carlo, L = {}".format(projs[l]), lw=2, ms=10)
        plt.fill_between(dims, np.percentile(errors_mc[:, :, l], 10, axis=0), np.percentile(errors_mc[:, :, l], 90, axis=0), alpha=0.2)
    plt.plot(dims, errors_approx.mean(axis=0), next(marker), label="Our method", lw=2, ms=10)
    plt.fill_between(dims, np.percentile(errors_approx, 10, axis=0), np.percentile(errors_approx, 90, axis=0), alpha=0.2)
    plt.xlabel("dimension")
    plt.ylabel("approximation error")
    plt.xscale('log')
    plt.legend()
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figures/comparison_errors.pdf')

    # Plot average computation time vs. dimension
    marker = itertools.cycle(('-x', '-o', '-^', '-*', '--.'))
    plt.figure(figsize=[5, 4])
    for l in range(len(projs)):
        plt.plot(dims, time_mc[:, :, l].mean(axis=0), next(marker), lw=2, ms=10)
    plt.plot(dims, time_approx.mean(axis=0), next(marker), lw=2, ms=10)
    plt.xlabel("dimension")
    plt.ylabel("time (in sec)")
    plt.xscale('log')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figures/comparison_time.pdf')


if __name__ == "__main__":
    # Create directory that will contain the results
    dirname = os.path.join("figures")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Reproduce Figures 1(a) and 1(b)
    run_indep()

    # Reproduce Figures 1(c) and 1(d)
    run_ar()

    # Reproduce Figure 2
    run_comparison()
