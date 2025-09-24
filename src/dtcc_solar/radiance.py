import os
import math
import subprocess
import numpy as np

from typing import Tuple
from io import StringIO
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.utils import SkyType


def epw_to_wea(epw_path: str, epw2wea_path: str) -> str | None:
    """
    Convert an EPW file to a WEA file using Radiance's epw2wea utility.

    Parameters:
    - epw_path: Path to the input .epw file

    Returns:
    - Path to the generated .wea file if successful, None otherwise
    """

    if not os.path.exists(epw_path):
        error(f"EPW file not found at {epw_path}")
        return None

    base = os.path.splitext(epw_path)[0]
    wea_path = base + ".wea"

    if os.path.exists(wea_path):
        info(f"WEA file already exists. Skipping conversion.")
        return wea_path

    command = [epw2wea_path, epw_path, wea_path]

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            error("Error during epw2wea conversion:")
            error(result.stderr)
            return None
        else:
            info(f"Conversion successful. WEA file saved to:")
            info(f"{wea_path}")
            return wea_path

    except FileNotFoundError:
        error(f"Error: '{epw2wea_path}' not found. Is Radiance installed? Check path.")
        return None


def run_subprocess(command: list) -> subprocess.CompletedProcess | None:
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result
    except FileNotFoundError:
        print(f"Error: '{command[0]}' not found. Is Radiance installed? Check path.")
        return None


def get_matrix(result):

    if result is None:
        return None

    if result.returncode != 0:
        print("Error during gendaymtx execution:")
        print(result.stderr)
    else:
        lines = result.stdout.splitlines()

        # Find where numeric data starts (first line beginning with a digit)
        start_idx = next(i for i, l in enumerate(lines) if l.strip() and l[0].isdigit())
        matrix = np.loadtxt(StringIO("\n".join(lines[start_idx:])))

        # Parse header
        meta = {}
        for line in lines:
            if "=" in line:
                key, val = line.split("=", 1)
                meta[key.strip()] = (
                    int(val.strip()) if val.strip().isdigit() else val.strip()
                )

        nrows = int(meta["NROWS"])
        ncols = int(meta["NCOLS"])
        ncomp = int(meta["NCOMP"])

        # Reshape full array
        full_matrix = matrix.reshape((nrows, ncols, ncomp))

        # Only the Tregenza skydome has the ground as an extra patch.
        if nrows == 146:
            # Drop the ground patch (last row in Tregenza, NOTE: not in Reinhart!)
            full_matrix = full_matrix[:-1, :, :]

        # Reduce to single irradiance channel (take first component)
        irr_matrix = full_matrix[:, :, 0]

    return irr_matrix


def calc_radiance_matrices(epw_path: str, sky_type: SkyType, rad_path: str):

    info("-----------------------------------------------------")

    epw2wea_path = rad_path + "epw2wea"
    wea_file = epw_to_wea(epw_path, epw2wea_path)
    if not wea_file:
        error("Failed to convert EPW to WEA.")
        return None

    gendaymtx_path = rad_path + "gendaymtx"

    info("Computing radiance sky matrix from WEA file")
    a = os.path.exists(gendaymtx_path)
    info(f"Radiance gendaymtx found: {a}")

    sky_command = get_sky_command(wea_file, sky_type, gendaymtx_path)
    result = run_subprocess(sky_command)
    sky_matrix = get_matrix(result)

    sun_command = get_sun_command(wea_file, sky_type, gendaymtx_path)
    result = run_subprocess(sun_command)
    sun_matrix = get_matrix(result)

    info("Radiance matrix shapes:")
    info(f"Sky matrix shape from Radiance : {sky_matrix.shape}")
    info(f"Sun matrix shape from Radiance : {sun_matrix.shape}")
    info("-----------------------------------------------------")

    tot_matrix = sky_matrix + sun_matrix

    return sky_matrix, sun_matrix, tot_matrix


def get_sun_command(wea_file, sky_type: SkyType, gendaymtx_path: str) -> list[str]:
    # Compute sun matrix command for gendaymtx
    # -d = direct (sun only)
    # -m = method (2 for Reinhart 580)

    if sky_type == SkyType.REINHART_578:
        return [gendaymtx_path, "-O1", "-d", "-m", "2", wea_file]

    else:  # default is Tregenza sky discretisation
        return [gendaymtx_path, "-O1", "-d", wea_file]


def get_sky_command(wea_file, sky_type: SkyType, gendaymtx_path: str) -> list[str]:

    # Compute sky matrix command for gendaymtx
    # -s = sky
    # -c = color  (1 1 1 for RGB channels to get a white sky)
    # -m = method (2 for Reinhart 580)

    if sky_type == SkyType.REINHART_578:
        return [gendaymtx_path, "-O1", "-s", "-m", "2", "-c", "1", "1", "1", wea_file]

    else:  # default is Tregenza sky discretisation
        return [gendaymtx_path, "-O1", "-s", "-c", "1", "1", "1", wea_file]
