from dtcc_solar.perez_complete_coeffs import calc_perez_coeffs
from dtcc_solar.perez import calculate_air_mass, compute_sky_brightness
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SolarParameters, DataSource
import matplotlib.pyplot as plt
import numpy as np


def plot_coeffs():
    """
    Example function to plot a simple 2D sine wave.
    """
    n = 100
    epsilons = np.linspace(1, 6.3, n)

    epsilons = [1.000, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200]

    zenith = np.pi / 4  # Example zenith angle
    plt.figure("Simulated Perez Coefficients", figsize=(12, 6))

    delta_low = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2])
    delta_high = np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5])
    delta_mid = delta_low + (delta_high - delta_low) / 2.0

    delta_dict = {
        "low": delta_low,
        "mid": delta_mid,
        "high": delta_high,
    }

    for key, deltas in delta_dict.items():
        As, Bs, Cs, Ds, Es = [], [], [], [], []

        for i, epsilon in enumerate(epsilons):
            delta = deltas[i]
            A, B, C, D, E = calc_perez_coeffs(epsilon, delta, zenith)
            As.append(A)
            Bs.append(B)
            Cs.append(C)
            Ds.append(D)
            Es.append(E)

        # plt.plot(range(1, len(epsilons) + 1), As, label=f"A with delta={key}")
        plt.plot(range(1, len(epsilons) + 1), Bs, label=f"B with delta={key}")
        # plt.plot(range(1, len(epsilons) + 1), Cs, label=f"C with delta={key}")
        # plt.plot(range(1, len(epsilons) + 1), Ds, label=f"D with delta={key}")
        # plt.plot(range(1, len(epsilons) + 1), Es, label=f"E with delta={key}")

    plt.xlabel(f"Epsilon (Sky Clearness)")
    plt.ylabel("Coefficients")
    plt.title("Simulated Perez Coefficients")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_air_mass():
    """
    Example function to plot air mass as a function of zenith angle.
    """
    n = 100
    zenith_angles = np.linspace(0, np.pi / 2, n)  # Zenith angles from 0 to 90 degrees

    air_masses = [calculate_air_mass(z) for z in zenith_angles]

    deltas = compute_sky_brightness(200, air_masses)

    plt.figure("Air Mass vs Zenith Angle", figsize=(12, 6))
    plt.plot(np.degrees(zenith_angles), air_masses, label="Air Mass", color="blue")
    plt.xlabel("Zenith Angle (degrees)")
    plt.ylabel("Air Mass")
    plt.title("Air Mass as a Function of Zenith Angle")
    plt.legend()
    plt.grid(True)


def plot_weather_data():

    path_lnd = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"

    long_lnd = 0.12
    lat_lnd = 51.5

    p = SolarParameters(
        file_name="",
        weather_file=path_lnd,
        start_date="2019-01-01 00:00:00",
        end_date="2019-02-01 23:00:00",
        longitude=long_lnd,
        latitude=lat_lnd,
        data_source=DataSource.epw,
        sun_analysis=True,
        sky_analysis=True,
    )

    sunpath = Sunpath(p, 1)
    sunc = sunpath.sunc

    plt.figure("Dni and sythetic Dni data", figsize=(12, 6))
    plt.plot(range(len(sunc.dni)), sunc.dni, label="DNI", color="blue")
    plt.plot(
        range(len(sunc.synth_dni)),
        sunc.synth_dni,
        label="SynthDNI",
        color="red",
        linestyle="-.",
    )
    plt.xlabel("Sun hours")
    plt.ylabel("W/m2")
    plt.title("DNI and DHI Data")
    plt.legend()
    plt.grid(True)

    plt.figure("Dhi and Dni data", figsize=(12, 6))
    plt.plot(range(len(sunc.dhi)), sunc.dhi, label="DHI", color="blue")
    plt.plot(
        range(len(sunc.synth_dhi)),
        sunc.synth_dhi,
        label="SynthDHI",
        color="red",
        linestyle="-.",
    )
    plt.xlabel("Sun hours")
    plt.ylabel("W/m2")
    plt.title("DNI and DHI Data")
    plt.legend()
    plt.grid(True)


if __name__ == "__main__":
    # plot_air_mass()
    plot_weather_data()
    plt.show()
