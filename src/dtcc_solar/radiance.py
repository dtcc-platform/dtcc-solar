import subprocess
import os
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


def load_radiance(wea_file):

    info("Computing radiance sky matrix from WEA file")
    a = os.path.exists("/usr/local/radiance/bin/gendaymtx")
    info(f"Radiance gendaymtx found: {a}")

    command = ["/usr/local/radiance/bin/gendaymtx", "-v", "-O1", "-d", wea_file]
    # command = [gdm, "-v", "-O1", "-d", wea_file, "-o", "sky_matrix"]

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    info("Radiance gendaymtx executed via subprocess.run")

    print(type(result.stdout))

    # matrix = np.loadtxt(StringIO(result.stdout))

    print("Radiance gendaymtx output:")
    if result.returncode != 0:
        print("Error during gendaymtx execution:")
        print(result.stderr)
    else:
        print(result.stdout)

    # print(matrix.shape)


def run_radiance(wea_file: str, output_dir: str = "."):
    """
    Run Radiance gendaymtx to generate Reinhart sky matrices for diffuse and direct components.
    Saves two files:
        - reinhart_diffuse.mtx
        - reinhart_direct.mtx
    """
    gdm_path = "/usr/local/radiance/bin/gendaymtx"
    if not os.path.exists(gdm_path):
        error(f"Radiance gendaymtx not found at: {gdm_path}")
        return

    info("Computing Radiance Reinhart sky matrices from WEA file...")

    # Output paths
    diffuse_path = os.path.join(output_dir, "reinhart_diffuse.mtx")
    direct_path = os.path.join(output_dir, "reinhart_direct.mtx")

    # Command for diffuse matrix (sky only)
    cmd_diffuse = [gdm_path, "-r", "-m", "1", "-d", "-O1", "-skyonly", wea_file]
    # Command for direct sun matrix
    cmd_direct = [gdm_path, "-r", "-m", "1", "-D", "-O1", wea_file]

    # Run diffuse
    try:
        with open(diffuse_path, "w") as f:
            result = subprocess.run(
                cmd_diffuse, stdout=f, stderr=subprocess.PIPE, text=True
            )
        if result.returncode != 0:
            error(f"Diffuse matrix error:\n{result.stderr}")
        else:
            info(f"Diffuse matrix saved to {diffuse_path}")
    except Exception as e:
        error(f"Failed to run diffuse gendaymtx: {e}")

    # Run direct
    try:
        with open(direct_path, "w") as f:
            result = subprocess.run(
                cmd_direct, stdout=f, stderr=subprocess.PIPE, text=True
            )
        if result.returncode != 0:
            error(f"Direct matrix error:\n{result.stderr}")
        else:
            info(f"Direct matrix saved to {direct_path}")
    except Exception as e:
        error(f"Failed to run direct gendaymtx: {e}")
