import os
import time

from dtcc_solar.utils import OutputCollection, SolarParameters, DataSource, SunApprox
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from dtcc_solar.colors import create_data_dict
from dtcc_solar.utils import get_sub_mesh, get_sub_face_mask

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

    engine.set_face_mask([0.45, 0.55], [0.45, 0.55])
    engine.subdivide_masked_mesh(3.0)
    engine.run_analysis(p, sunpath, outputc)

    if p.display:
        data_dict_1, data_dict_2 = create_data_dict(outputc, engine.face_mask)
        mesh_1, mesh_2 = engine.split_mesh_by_face_mask()
        viewer = Viewer()
        viewer.build_sunpath_diagram(sunpath, p)
        viewer.add_mesh("Analysed mesh", mesh=mesh_1, data=data_dict_1)
        viewer.add_mesh("Shading mesh", mesh=mesh_2, data=data_dict_2)
        viewer.show()


if __name__ == "__main__":
    set_log_level("INFO")
    inputfile_S = "../../../data/models/CitySurfaceS.stl"
    inputfile_M = "../../../data/models/CitySurfaceM.stl"
    inputfile_L = "../../../data/models/CitySurfaceL.stl"

    path = "../../../data/weather/"

    # inputfile_L = "../../../data/models/lozenets.stl"
    # inputfile_L = "../../../data/models/City136kSoft.stl"

    lnd_clm = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
    lnd_epw = "GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw"
    gbg_clm = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.clm"
    gbg_epw = "SWE_VG_Gothenburg-Landvetter.AP.025260_TMYx.2007-2021.epw"
    sth_clm = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.clm"
    sth_epw = "SWE_ST_Stockholm.Arlanda.AP.024600_TMYx.2007-2021.epw"

    # Gothenburg
    p_1 = SolarParameters(
        file_name=inputfile_L,
        weather_file=path + gbg_clm,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        longitude=11.97,
        latitude=57.71,
        data_source=DataSource.smhi,
        sun_analysis=True,
        sky_analysis=True,
        sun_approx=SunApprox.group,
    )

    # Stockholm
    p_2 = SolarParameters(
        file_name=inputfile_L,
        weather_file=path + sth_clm,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        longitude=18.063,
        latitude=59.33,
        data_source=DataSource.clm,
        sun_analysis=True,
        sky_analysis=True,
        sun_approx=SunApprox.quad,
    )

    # Rio de Janeiro
    p_3 = SolarParameters(
        file_name=inputfile_L,
        start_date="2019-01-01 00:00:00",
        end_date="2019-12-31 00:00:00",
        longitude=-43.19,
        latitude=-22.90,
        data_source=DataSource.meteo,
        sun_analysis=True,
        sky_analysis=True,
        sun_approx=SunApprox.group,
    )

    run_script(p_3)
