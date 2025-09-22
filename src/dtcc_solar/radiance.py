import subprocess
import os
from unittest import result
import numpy as np
from io import StringIO
from dtcc_solar.logging import info, debug, warning, error


def epw_to_wea(epw_path: str) -> str | None:
    """
    Convert an EPW file to a WEA file using Radiance's epw2wea utility.

    Parameters:
    - epw_path: Path to the input .epw file

    Returns:
    - Path to the generated .wea file if successful, None otherwise
    """

    if not os.path.exists(epw_path):
        print(f"Error: EPW file not found at {epw_path}")
        return None

    base = os.path.splitext(epw_path)[0]
    wea_path = base + ".wea"

    if os.path.exists(wea_path):
        print(f"WEA file already exists at {wea_path}. Skipping conversion.")
        return wea_path

    epw2wea = "/usr/local/radiance/bin/epw2wea"  # Adjust this if Radiance is installed elsewhere

    command = [epw2wea, epw_path, wea_path]

    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            print("Error during epw2wea conversion:")
            print(result.stderr)
            return None
        else:
            print(f"Conversion successful. WEA file saved to: {wea_path}")
            return wea_path

    except FileNotFoundError:
        print(f"Error: '{epw2wea}' not found. Is Radiance installed? Check path.")
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
        print("Matrix shape:", matrix.shape)

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
        print("Full shape:", full_matrix.shape)  # (146, 8760, 3)

        # Drop the ground patch (last row)
        drop_matrix = full_matrix[:-1, :, :]
        print("Sky-only shape:", drop_matrix.shape)  # (145, 8760, 3)

        # Reduce to single irradiance channel (take first component)
        irr_matrix = drop_matrix[:, :, 0]
        print("Irradiance shape:", irr_matrix.shape)  # (145, 8760)

    return irr_matrix


def compute_radiance_matrices(epw_path: str):

    wea_file = epw_to_wea(epw_path)
    if not wea_file:
        error("Failed to convert EPW to WEA.")
        return None

    info("Computing radiance sky matrix from WEA file")
    a = os.path.exists("/usr/local/radiance/bin/gendaymtx")
    info(f"Radiance gendaymtx found: {a}")

    path = "/usr/local/radiance/bin/gendaymtx"

    # Compute sky matrix (s = sky)
    sky_command = [path, "-O1", "-s", wea_file]
    result = run_subprocess(sky_command)
    sky_matrix = get_matrix(result)

    # Compute sun matrix (d = direct)
    sun_command = [path, "-O1", "-d", wea_file]
    result = run_subprocess(sun_command)
    sun_matrix = get_matrix(result)

    tot_matrix = sky_matrix + sun_matrix

    return sky_matrix, sun_matrix, tot_matrix
