import numpy as np
import matplotlib.pyplot as plt


class SunPatchWeighting:
    """
    Radiance-style sun patch weighting function.

    The weighting function is defined as:
        w(θ) = 1 / (1.002 - cos(θ))
    where θ is the angle from the sun direction.

    This function is used to weight the contribution of sky patches
    based on their angular distance from the sun direction.
    """

    def __init__(self):
        pass

    def _w(self, theta_deg):
        theta = np.radians(theta_deg)
        dprod = np.cos(theta)
        return 1.0 / (1.002 - dprod)

    def plot(self, angles_deg: np.ndarray):
        weights = self._w(angles_deg)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(angles_deg, weights, label=r"$w(\theta) = 1/(1.002 - \cos\theta)$")
        # plt.yscale("log")
        plt.xlabel("Angle θ from sun direction (degrees)")
        plt.ylabel("Weight (log scale)")
        plt.title("Radiance-style sun patch weighting function")
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    angles_deg = np.linspace(0, 20, 500)
    pw = SunPatchWeighting()
    pw.plot(angles_deg)
