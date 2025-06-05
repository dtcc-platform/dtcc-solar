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
