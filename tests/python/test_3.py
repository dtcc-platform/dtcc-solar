import os
import numpy as np
import pytest
from datetime import datetime
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.synthetic_data import synthetic_epw_df
from dtcc_solar.utils import SolarParameters, SunCollection
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.reinhart2 import ReinhartM2
from dtcc_solar.reinhart4 import ReinhartM4
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_core.model import Mesh


def test_tregenza():
    """
    Test that the Tregenza skydome has the correct number of patches
    and the total solid angle is approximately 2π steradians.
    """
    dome = Tregenza()

    # Assert patch count
    assert dome.patch_counter == 145

    # Assert direction vector count matches
    assert len(dome.ray_dirs) == 145

    # Check normalization of direction vectors
    norms = np.linalg.norm(dome.ray_dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Total solid angle ≈ 2π (hemisphere)
    total_solid_angle = np.sum(dome.solid_angles)
    assert np.isclose(total_solid_angle, 2 * np.pi, rtol=1e-3)


def test_reinhart_m2():
    """
    Test that Reinhart M2 skydomes have the expected number of patches
    and the total solid angle sums to approximately 2π.
    """
    dome = ReinhartM2()

    # Assert patch count
    assert dome.patch_counter == 578

    # Assert direction vector count matches
    assert len(dome.ray_dirs) == 578

    # Check normalization of direction vectors
    norms = np.linalg.norm(dome.ray_dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Total solid angle ≈ 2π (hemisphere)
    total_solid_angle = np.sum(dome.solid_angles)
    assert np.isclose(total_solid_angle, 2 * np.pi, rtol=1e-3)


def test_reinhart_m4():
    """
    Test that Reinhart M4 skydomes have the expected number of patches
    and the total solid angle sums to approximately 2π.
    """
    dome = ReinhartM4()

    # Assert patch count
    assert dome.patch_counter == 2305

    # Assert direction vector count matches
    assert len(dome.ray_dirs) == 2305

    # Check normalization of direction vectors
    norms = np.linalg.norm(dome.ray_dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Total solid angle ≈ 2π (hemisphere)
    total_solid_angle = np.sum(dome.solid_angles)
    assert np.isclose(total_solid_angle, 2 * np.pi, rtol=1e-3)


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Raytracing test started -----------------------")
    set_log_level("INFO")

    test_tregenza()
    test_reinhart_m2()
    test_reinhart_m4()
