import numpy as np
import pandas as pd
import os
import time

from dtcc_solar.utils import ColorBy, OutputCollection, SolarParameters, DataSource
from dtcc_solar.utils import SunApprox
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.sundome import SunDome
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.colors import color_mesh, create_data_dict
from dtcc_solar.sungroups import SunGroups

from dtcc_model import Mesh
from dtcc_io import meshes
from pprint import pp


def run_script(solar_parameters: SolarParameters):
    os.system("clear")
    print("-------- Solar Analysis Started -------")

    p = solar_parameters
    mesh = meshes.load_mesh(p.file_name)
    engine = SolarEngine(mesh)
    sunpath = Sunpath(p, engine.sunpath_radius)
    outputc = OutputCollection()

    engine.run_analysis(p, sunpath, outputc)

    if p.display:
        data_dict = create_data_dict(outputc)
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath)
        viewer.add_mesh("City mesh", mesh=mesh, data=data_dict)
        viewer.show()


if __name__ == "__main__":
    set_log_level("INFO")
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    weather_file_clm = (
        "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    )

    # Iterative solar analysis
    p_1 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_dn,
        sun_analysis=True,
        sky_analysis=True,
        sun_approx=SunApprox.group,
    )

    # Iterative solar analysis
    p_2 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.irradiance_dn,
        sun_analysis=True,
        sky_analysis=True,
        sun_approx=SunApprox.quad,
    )

    run_script(p_1)
