import numpy as np
import matplotlib.pyplot as plt
from dtcc_solar.utils import SkyResults


def sub_plots_2d(res: SkyResults, t1, t2):

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    axs = axs.flatten()  # Flatten to simplify indexing

    subplot2d(axs[0], res.relative_luminance, t1, t2, title="Relative Luminance")
    subplot2d(axs[1], res.matrix, t1, t2, title="Sky Vector Matrix")

    plt.tight_layout()


def sub_plot_dict(data: dict, t1, t2):

    n = len(data.keys())
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(16, 10))
    axs = axs.flatten()  # Flatten to simplify indexing
    counter = 0

    t2 = min(t2, len(next(iter(data.values()))))

    for key, value in data.items():
        subplot1d(axs[counter], value, t1, t2, name=key)
        counter += 1

    plt.tight_layout()


def subplot2d(ax, data: np.ndarray, t1=0, t2=5, title="title"):
    """
    Plot a slice of 2D data on a given Axes subplot.
    """
    n_patches, n_timesteps = data.shape
    upper = min(t2, n_timesteps)
    timestep_indices = list(range(t1, upper))

    for i in timestep_indices:
        F_values = data[:, i]
        ax.plot(range(n_patches), F_values, label=f"Sun #{i}")

    ax.set_xlabel("Sky Patch Index")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def subplot1d(ax, data: np.ndarray, t1, t2, name="title"):
    """
    Plot a slice of 2D data on a given Axes subplot.
    """
    data_cropped = data[t1:t2]
    n = len(data_cropped)
    ax.plot(range(t1, t2), data_cropped, label=name)

    ax.set_xlabel("Sun index")
    ax.set_ylabel("Value")
    ax.set_title(name)
    ax.grid(True)
    ax.legend()


def plot_2d_data(data: np.ndarray, t1=0, t2=5, title="title"):

    n_patches, n_timesteps = data.shape
    upper = min(t2, n_timesteps)
    timestep_indices = list(range(t1, upper))

    print(f"Plotting {len(timestep_indices)} timesteps from {t1} to {upper}.")

    plt.figure(figsize=(12, 6))

    for i in timestep_indices:
        F_values = data[:, i]
        plt.plot(range(n_patches), F_values, label=f"Sun #{i}")

    plt.xlabel("Sky Patch Index")
    plt.ylabel("Relative Luminance F")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_coeffs_dict(data: dict, t1=0, t2=5, title="title"):
    """
    Plot a slice of coefficient data from a dictionary.

    Parameters:
    - data: dict with keys ["A", "B", "C", "D", "E"], each mapping to a 1D array of values
    - data_name: label for y-axis
    - t1, t2: time index range for plotting
    - title: plot title
    """

    plt.figure("Perez Coefficients", figsize=(12, 6))

    for name in data.keys():
        values = data[name][t1:t2]
        plt.plot(range(t1, t1 + len(values)), values, label=name)

    plt.xlabel(f"Sun indices from {t1} to {t2}")
    plt.ylabel("coefficients")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_debug_1(arr1: np.array, arr2: np.array):

    plt.figure(figsize=(12, 6))
    for i in range(len(arr1)):
        plt.plot(arr1[i, :], arr2[i, :])

    plt.xlabel("gamma")
    plt.ylabel("Rvs")
    plt.grid(True)
    plt.tight_layout()


def plot_debug_2(arr1: np.array, arr2: np.array):

    plt.figure(figsize=(12, 6))
    plt.plot(arr1, arr2)

    plt.xlabel("sun_zenith")
    plt.ylabel("Rvs_max")
    plt.grid(True)
    plt.tight_layout()


def show_plot():
    """
    Show the current plot.
    """
    plt.show()
