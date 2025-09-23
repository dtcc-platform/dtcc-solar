import os
import math
import subprocess
import numpy as np

from typing import Tuple
from io import StringIO
from dtcc_solar.logging import info, debug, warning, error
from dtcc_solar.utils import SkyType


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


def calc_radiance_matrices(epw_path: str, sky_type: SkyType):

    wea_file = epw_to_wea(epw_path)
    if not wea_file:
        error("Failed to convert EPW to WEA.")
        return None

    info("Computing radiance sky matrix from WEA file")
    a = os.path.exists("/usr/local/radiance/bin/gendaymtx")
    info(f"Radiance gendaymtx found: {a}")

    sky_command = get_sky_command(wea_file, sky_type)
    result = run_subprocess(sky_command)
    sky_matrix = get_matrix(result)

    # Compute sun matrix (d = direct)
    sun_command = get_sun_command(wea_file, sky_type)
    result = run_subprocess(sun_command)
    sun_matrix = get_matrix(result)

    tot_matrix = sky_matrix + sun_matrix

    return sky_matrix, sun_matrix, tot_matrix


def get_sun_command(wea_file, sky_type: SkyType) -> list[str]:
    # Compute sun matrix command for gendaymtx
    # -d = direct (sun only)
    # -m = method (2 for Reinhart 580)

    path = "/usr/local/radiance/bin/gendaymtx"

    if sky_type == SkyType.REINHART_580:
        return [path, "-O1", "-d", "-m", "2", wea_file]

    else:  # default is Tregenza sky discretisation
        return [path, "-O1", "-d", wea_file]


def get_sky_command(wea_file, sky_type: SkyType) -> list[str]:

    # Compute sky matrix command for gendaymtx
    # -s = sky
    # -c = color  (1 1 1 for RGB channels to get a white sky)
    # -m = method (2 for Reinhart 580)

    path = "/usr/local/radiance/bin/gendaymtx"

    if sky_type == SkyType.REINHART_580:
        return [path, "-O1", "-s", "-m", "2", "-c", "1", "1", "1", wea_file]

    else:  # default is Tregenza sky discretisation
        return [path, "-O1", "-s", "-c", "1", "1", "1", wea_file]


# -----------------------------------------------------------------------#


def get_tregenza_from_rad(rhsubdiv: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduce Radiance's Reinhart/Tregenza sky patch centers and solid angles,
    exactly as implemented in gendaymtx::rh_init().

    Returns
    -------
    ray_dirs : (145, 3) float
        Unit vectors for the 145 *sky* patch centers (X east, Y north, Z up),
        ordered like Radiance (row 1 .. 7, then the zenith cap last).
    solid_angles : (145,) float
        Patch solid angles in steradians; sum ≈ 2π.

    Notes
    -----
    - Radiance coordinate system: X east, Y north, Z up.
    - Azimuth is measured east of North (i.e., from +Y towards +X).
    - We drop the ground patch Radiance keeps at index 0.
    - For rhsubdiv=1 this yields the classic Tregenza layout: rows
      with 30,30,24,24,18,12,6 patches plus 1 zenith cap.
    """
    # Radiance gendaymtx uses these row counts (for rhsubdiv=1)
    tnaz = np.array([30, 30, 24, 24, 18, 12, 6], dtype=int)
    NROW = len(tnaz)

    # Effective “row height” in radians (alpha in gendaymtx)
    alpha = (math.pi / 2.0) / (NROW * rhsubdiv + 0.5)

    ray_dirs = []
    solid_angles = []

    # ---- “Normal” rows (exclude ground and zenith cap) ----
    for i in range(NROW * rhsubdiv):
        # Row center altitude (radians from horizon)
        ralt = alpha * (i + 0.5)
        # Patches in this row
        ninrow = int(tnaz[i // rhsubdiv] * rhsubdiv)

        # Solid angle per patch in this row
        # dom = 2π * (sin(alt_top) - sin(alt_bottom)) / ninrow
        dom_row = 2.0 * math.pi * (math.sin(alpha * (i + 1)) - math.sin(alpha * i))
        dom_patch = dom_row / ninrow

        # Centers at uniform azimuthal spacing
        for j in range(ninrow):
            azi = (
                2.0 * math.pi * j / float(ninrow)
            )  # 0 at +Y (north), increasing towards +X (east)
            ca = math.cos(ralt)
            sx = ca * math.sin(azi)  # X east
            sy = ca * math.cos(azi)  # Y north
            sz = math.sin(ralt)  # Z up
            ray_dirs.append([sx, sy, sz])
            solid_angles.append(dom_patch)

    # ---- Zenith cap (last sky patch) ----
    # Radiance sets zenith cap solid angle: 2π*(1 - cos(alpha*0.5))
    zen_dom = 2.0 * math.pi * (1.0 - math.cos(alpha * 0.5))
    ray_dirs.append([0.0, 0.0, 1.0])
    solid_angles.append(zen_dom)

    ray_dirs = np.asarray(ray_dirs, dtype=float)
    solid_angles = np.asarray(solid_angles, dtype=float)

    # Sanity checks (sum of sky dom ≈ 2π, and lengths are unit)
    # (Leave them enabled during development; you can remove later.)
    assert ray_dirs.shape == (145, 3)
    assert solid_angles.shape == (145,)
    assert np.allclose(np.linalg.norm(ray_dirs, axis=1), 1.0, atol=1e-12)
    assert abs(solid_angles.sum() - 2.0 * math.pi) < 1e-8

    return ray_dirs, solid_angles
