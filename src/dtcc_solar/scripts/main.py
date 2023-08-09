import numpy as np
import pandas as pd
import time
import os
import trimesh
import argparse
import sys

from dtcc_solar import utils
from dtcc_solar import data_io 
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.sunpath import SunpathMesh
from dtcc_solar.viewer import Viewer
from dtcc_solar.utils import ColorBy, AnalysisType, Parameters, DataSource
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
from dtcc_solar.utils import Sun, Output
from dtcc_solar import weather_data as wd
from dtcc_solar.viewer import Colors

from pprint import pp
from pprint import pprint
from typing import List, Dict

import dtcc_solar.smhi_data as smhi
import dtcc_solar.meteo_data as meteo  
import dtcc_solar.epw_data as epw
import dtcc_solar.clm_data as clm

def register_args(args):    
    
    default_path = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceL.stl'
    parser = argparse.ArgumentParser(description='Parameters to run city solar analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--analysis'    , type=int  , metavar='', default=1, help=' sun_raycasting = 1, sky_raycasting = 2, com_raycasting = 3, dome_raycasting = 4')
    parser.add_argument('-lat', '--latitude'  , type=float, metavar='', default=51.5 , help='Latitude for location of analysis')
    parser.add_argument('-lon', '--longitude' , type=float, metavar='', default=-0.12, help='Longitude for location of analysis')
    parser.add_argument('-f', '--inputfile'   , type=str  , metavar='', default=default_path, help='Filename (incl. path) for city mesh (*.stl, *.vtu)')
    parser.add_argument('-sd', '--start_date' , type=str  , metavar='', default='2019-03-30 07:00:00', help='Start date for iterative analysis')
    parser.add_argument('-ed', '--end_date'   , type=str  , metavar='', default='2019-03-30 21:00:00', help='End date for iterative analysis')
    parser.add_argument('-disp', '--display'  , type=int  , metavar='', default=1, help='Display results with pyglet')
    parser.add_argument('-pd', '--prep_disp'  , type=int  , metavar='', default=1, help='Preproscess colors and other graphic for display')
    parser.add_argument('-ds', '--data_source', type=int  , metavar='', default=3, help='Enum for data source. 1 = SMHI, 2 = Open Meteo, 3 = Clm file, 4 = Epw file')
    parser.add_argument('-c', '--colorby'     , type=int  , metavar='', default=2, help='Colo_by: face_sun_angle =1, face_sun_angle_shadows = 2, shadows = 3 , irradiance_direct_normal = 4, irradiance_direct_horizontal = 5, irradiance_diffuse = 6, irradiance_all = 6')
    parser.add_argument('-e', '--export'      , type=bool , metavar='', default=True, help='Export data')
    parser.add_argument('-ep', '--exportpath' , type=str  , metavar='', default='./data/dataExport.txt', help='Path for data export of type *.txt')
    parser.add_argument('-wf', '--w_file'     , type=str  , metavar='', default='./data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm', help='Weather data file to be uploaded by the user')
    new_args = parser.parse_args(args)
    return new_args

def print_args(args):
    for arg in vars(args):
        print(arg, '\t', getattr(args, arg))
    print("----------------------------------------")    
        
    
def export(p:Parameters, city_results:Results, exportpath:str):
    if p.color_by == ColorBy.face_sun_angle: 
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_sun_angle_shadows: 
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)    
    elif p.color_by == ColorBy.face_irradiance_dn: 
        data_io.print_list(city_results.get_face_irradiance(), exportpath)
    elif p.color_by == ColorBy.face_shadows: 
        data_io.print_list(city_results.get_face_in_sun(), exportpath)    



###############################################################################################################################

def run_script(command_line_args):

    clock1 = time.perf_counter()
    os.system('clear')
    print("-------- Solar Analysis Started -------")

    args = register_args(command_line_args)
    print(command_line_args)

    #Convert command line input to enums and data formated for the analysis
    p = Parameters(args.analysis,   args.inputfile,   args.latitude,    args.longitude, 
                   args.prep_disp,  args.display,     args.data_source, args.colorby,  
                   args.export,     args.start_date,  args.end_date,    args.w_file)    
    
    print_args(args)

    city_mesh = trimesh.load_mesh(p.file_name)
    solar_engine = SolarEngine(city_mesh)
    sunpath = Sunpath(p.latitude, p.longitude, solar_engine.sunpath_radius)
    suns = sunpath.create_suns(p)
    results = Results(suns, len(city_mesh.faces)) 

    #Execute analysis        
    if  (p.a_type == AnalysisType.sun_raycasting):    
        solar_engine.sun_raycasting(suns, results)
    elif(p.a_type == AnalysisType.sky_raycasting):    
        solar_engine.sky_raycasting(suns, results)
    elif(p.a_type == AnalysisType.com_raycasting):    
        solar_engine.sun_raycasting(suns, results)    
        solar_engine.sky_raycasting(suns, results) 

    results.calc_accumulated_results()
    results.calc_average_results()

    #Get geometry for the sunpath and current sun position
    if(p.prepare_display):

        viewer = Viewer()
        colors = Colors()

        # Create sunpath so that the solar postion are given a context in the 3D visualisation
        sunpath_mesh = SunpathMesh(solar_engine.sunpath_radius)
        sunpath_mesh.create_sunpath_diagram(suns, sunpath, solar_engine, colors)
        
        # Color city mesh and add to viewer
        colors.color_city_mesh(solar_engine.mesh, results.res_acum, p.color_by)
        viewer.add_meshes(solar_engine.mesh)

        # Add sunpath meshes to viewer
        viewer.add_meshes(sunpath_mesh.get_analemmas_meshes())
        viewer.add_meshes(sunpath_mesh.get_daypath_meshes())
        viewer.add_meshes(sunpath_mesh.get_sun_meshes())
        
        if(p.display):
            viewer.show()

    clock2 = time.perf_counter()
    print("Total computation time elapsed: " + str(round(clock2  - clock1,4)))
    print("----------------------------------------")
    return True                

if __name__ == "__main__":
    
    inputfile_S = '../../../data/models/CitySurfaceS.stl'
    inputfile_M = '../../../data/models/CitySurfaceM.stl'
    inputfile_L = '../../../data/models/CitySurfaceL.stl'

    other_file_to_run = '../../../data/models/new_file.stl'

    weather_file_clm = '../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'

    # Instant solar anaysis 
    args_1 = ['--inputfile', inputfile_L, 
              '--analysis', '1',
              '--start_date', '2019-03-30 09:00:00',
              '--end_date', '2019-03-30 09:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '2']

    # Instant sky analysis
    args_2 = ['--inputfile', inputfile_S, 
              '--analysis', '2',
              '--start_date', '2019-03-30 12:00:00',
              '--end_date', '2019-03-30 12:00:00', 
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '6']  
    
    # Instant combined analysis
    args_3 = ['--inputfile', inputfile_S, 
              '--analysis', '3',
              '--start_date', '2015-03-30 12:00:00',
              '--end_date', '2015-03-30 12:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm, 
              '--colorby', '7']

    # Iterative solar analysis 
    args_4 = ['--inputfile', inputfile_L, 
              '--analysis', '1',
              '--start_date', '2019-06-01 11:00:00',
              '--end_date', '2019-06-01 15:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm, 
              '--colorby', '4']

    # Iterative sky analysis 
    args_5 = ['--inputfile', inputfile_S, 
              '--analysis', '2',
              '--start_date', '2019-03-30 06:00:00',
              '--end_date', '2019-03-30 21:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '6']        
      
    # Iterative combined analysis
    args_6 = ['--inputfile', inputfile_S, 
              '--analysis', '3',
              '--start_date', '2019-03-30 06:00:00',
              '--end_date', '2019-03-30 21:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '7']   

    run_script(args_1)




