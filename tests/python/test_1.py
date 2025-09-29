import os
import numpy as np
import pytest
from datetime import datetime
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.synthetic_data import synthetic_epw_df
from dtcc_solar.utils import SolarParameters, SunCollection
from dtcc_solar.tregenza import Tregenza
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_core.model import Mesh


def test_1():
    """
    Smoke test: ensure 2-phase analysis runs end-to-end
    with a synthetic EPW dataset and produces
    expected array shapes.
    """

    # Example mesh: single triangle (flat surface at z=0)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(vertices=vertices, faces=faces)

    # Minimal analysis setup
    skydome = Tregenza()
    engine = SolarEngine(mesh)

    # To be continued...
    # Generate synthetic EPW dataframe for single day (hourly)
    # Create synthetic sunpath
    # Run 2 phase analysis
    # Assertions


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Raytracing test started -----------------------")
    set_log_level("INFO")
