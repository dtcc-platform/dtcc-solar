import numpy as np
import pandas as pd
import os

from dtcc_solar.utils import ColorBy, AnalysisType, SolarParameters, DataSource
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.results import Results
from dtcc_solar.sundome import SunDome
from dtcc_solar.skydome import SkyDome
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.colors import color_city_mesh

from dtcc_model import Mesh
from dtcc_io import meshes
from pprint import pp


def run_script(solar_parameters: SolarParameters):
    os.system("clear")
    print("-------- Solar Analysis Started -------")

    p = solar_parameters
    mesh = meshes.load_mesh(p.file_name)
    solar_engine = SolarEngine(mesh)
    sunpath = Sunpath(p, solar_engine.sunpath_radius)
    skydome = SkyDome(solar_engine.dome_radius, 10)
    sundome = SunDome(sunpath, solar_engine.horizon_z, 150, 20)
    results = Results(sunpath.suns, len(mesh.faces))

    solar_engine.run_analysis(p, sunpath, results, skydome, sundome)

    if p.display:
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath.suns, sunpath, sundome)
        colors = color_city_mesh(results.res_acum, p.color_by)
        viewer.add_mesh("City mesh", mesh=solar_engine.mesh, colors=colors)
        viewer.show()


if __name__ == "__main__":
    set_log_level("INFO")
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    weather_file_clm = (
        "../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    )

    # Instant solar anaysis
    p_1 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sun_raycasting,
        start_date="2019-03-30 09:00:00",
        end_date="2019-03-30 09:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_sun_angle_shadows,
    )

    # Instant sky analysis
    p_2 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sky_raycasting,
        start_date="2019-03-30 12:00:00",
        end_date="2019-03-30 12:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_di,
    )

    # Instant combined analysis
    p_3 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        a_type=AnalysisType.com_raycasting,
        start_date="2019-03-30 12:00:00",
        end_date="2019-03-30 12:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_tot,
    )

    # Iterative solar analysis
    p_4 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sun_raycasting,
        start_date="2019-06-01 11:00:00",
        end_date="2019-06-01 15:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_dn,
    )

    # Iterative sky analysis
    p_5 = SolarParameters(
        file_name=inputfile_S,
        weather_file=weather_file_clm,
        a_type=AnalysisType.sky_raycasting,
        start_date="2019-03-30 06:00:00",
        end_date="2019-03-30 21:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_di,
    )

    # Iterative combined analysis
    p_6 = SolarParameters(
        file_name=inputfile_L,
        weather_file=weather_file_clm,
        a_type=AnalysisType.com_raycasting,
        start_date="2019-03-30 06:00:00",
        end_date="2019-03-30 21:00:00",
        data_source=DataSource.clm,
        color_by=ColorBy.face_irradiance_tot,
    )

    run_script(p_3)
