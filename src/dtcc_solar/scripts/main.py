import numpy as np
import pandas as pd
import time
import os
import trimesh
import argparse

from dtcc_solar import utils
from dtcc_solar import data_io 
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.sunpath import SunpathMesh
from dtcc_solar.viewer import Viewer
from dtcc_solar.utils import ColorBy, AnalysisType, DataSource, Mode
from dtcc_solar.model import Model
from dtcc_solar.results import Results
from dtcc_solar.data_io import Parameters
from dtcc_solar.skydome import SkyDome
from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis
from dtcc_solar.multi_skydomes import MultiSkyDomes
from dtcc_solar.utils import Sun

from pprint import pp
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
    parser.add_argument('-d', '--one_date'    , type=str  , metavar='', default='2019-03-30 09:00:00', help='Date and time for instant analysis')
    parser.add_argument('-sd', '--start_date' , type=str  , metavar='', default='2019-03-30 07:00:00', help='Start date for iterative analysis')
    parser.add_argument('-ed', '--end_date'   , type=str  , metavar='', default='2019-03-30 21:00:00', help='End date for iterative analysis')
    parser.add_argument('-disp', '--display'  , type=int  , metavar='', default=1, help='Display results with pyglet')
    parser.add_argument('-pd', '--prep_disp'  , type=int  , metavar='', default=1, help='Preproscess colors and other graphic for display')
    parser.add_argument('-ds', '--data_source', type=int  , metavar='', default=3, help='Enum for data source. 1 = SMHI, 2 = Open Meteo, 3 = Clm file, 4 = Epw file')
    parser.add_argument('-c', '--colorby'     , type=int  , metavar='', default=2, help='Colo_by: face_sun_angle =1, face_sun_angle_shadows = 2, face_irradiance = 3 , face_shadows = 4, face_in_sky = 5, face_diffusion = 6')
    parser.add_argument('-e', '--export'      , type=bool , metavar='', default=True, help='Export data')
    parser.add_argument('-ep', '--exportpath' , type=str  , metavar='', default='./data/dataExport.txt', help='Path for data export of type *.txt')
    parser.add_argument('-m', '--mode'        , type=int  , metavar='', default=1, help='1 = single sun analysis, 2 = multiple sun analysis')
    parser.add_argument('-wf', '--w_file'     , type=str  , metavar='', default='./data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm', help='Weather data file to be uploaded by the user')
    new_args = parser.parse_args(args)
    return new_args

def print_args(args):
    for arg in vars(args):
        print(arg, '\t', getattr(args, arg))
    print("----------------------------------------")    
    
    
def run_iterative_sun(sun_analysis:SunAnalysis, suns:List[Sun]):     
    sun_analysis.execute_raycasting_iterative(suns)
    sun_analysis.set_city_mesh_out()
    
def run_iterative_sky(sky_analysis:SkyAnalysis, suns:List[Sun]):    
    sky_analysis.execute_raycasting_iterative(suns)
    sky_analysis.set_city_mesh_out()
    
def run_combined(sun_analysis:SunAnalysis, sky_analysis:SkyAnalysis, suns:List[Sun]):  
    sun_analysis.execute_raycasting_iterative(suns)    
    sky_analysis.execute_raycasting_iterative(suns)
    sun_analysis.set_city_mesh_out()
    
def run_sky_domes(sky_analysis:SkyAnalysis, suns:List[Sun]):  
    sky_analysis.execute_raycasting_some(suns)
    sky_analysis.set_city_mesh_out()
    sky_analysis.set_dome_mesh_out()

def export(p:Parameters, city_results:Results, exportpath:str):
    if p.color_by == ColorBy.face_sun_angle: 
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_sun_angle_shadows: 
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)    
    elif p.color_by == ColorBy.face_irradiance: 
        data_io.print_list(city_results.get_face_irradiance(), exportpath)
    elif p.color_by == ColorBy.face_shadows: 
        data_io.print_list(city_results.get_face_in_sun(), exportpath)    

def color_mesh(p:Parameters, city_results:Results, viewer:Viewer):
    if(p.a_type == AnalysisType.sun_raycasting):
        city_results.color_city_mesh_from_sun(p.color_by)
    elif(p.a_type == AnalysisType.sky_raycasting):
        city_results.color_city_mesh_from_sky(p.color_by)
    elif(p.a_type == AnalysisType.sky_raycasting_some):
        city_results.color_dome_mesh(p.color_by)
        viewer.add_dome_mesh(city_results.get_dome_mesh_out())
    elif(p.a_type == AnalysisType.com_raycasting):
        city_results.color_city_mesh_com_iterative(p.color_by)    


def get_weather_data(p:Parameters, suns:List[Sun]):
    if p.data_source == DataSource.smhi:
        suns = smhi.get_data_from_api_call(p.longitude, p.latitude, suns)
    if p.data_source == DataSource.meteo:
        suns = meteo.get_data_from_api_call(p.longitude, p.latitude, suns)
    elif p.data_source == DataSource.clm:
        suns = clm.import_weather_data_clm(suns, p.weather_file)
    elif p.data_source == DataSource.epw:
        suns = epw.import_weather_data_epw(suns,p.weather_file)
    return suns

def get_sun_and_sky(p:Parameters, sunpath:Sunpath):

    # Get date and time discretisation for single sun or multiple sun analysis
    if p.mode == Mode.single_sun:
        suns = utils.create_suns(p.one_date, p.one_date)                   
    elif p.mode == Mode.multiple_sun:
        suns = utils.create_suns(p.start_date, p.end_date)    
    
    clock1 = time.perf_counter()
    suns = get_weather_data(p, suns)                               
    clock2 = time.perf_counter()
    print("Weather data collection time elapsed: " + str(round(clock2  - clock1,4)))
    suns = sunpath.get_suns_positions(suns)                                        
    clock3 = time.perf_counter()
    print("Sun position collection time elapsed: " + str(round(clock3  - clock2,4)))
    
    return suns



###############################################################################################################################

def run_script(command_line_args):

    clock1 = time.perf_counter()
    os.system('clear')
    print("-------- Solar Analysis Started -------")

    args = register_args(command_line_args)
    print(command_line_args)

    #Convert command line input to enums and data formated for the analysis
    p = Parameters(args.analysis,   args.inputfile,   args.latitude, args.longitude, args.prep_disp, 
                   args.display,    args.data_source, args.colorby,  args.export,    args.one_date,   
                   args.start_date, args.end_date,    args.w_file,   args.mode)    
    
    print_args(args)

    city_mesh = trimesh.load_mesh(p.file_name)
    city_model = Model(city_mesh)
    city_results = Results(city_model)
    sunpath = Sunpath(p.latitude, p.longitude, city_model.sunpath_radius)
    skydome = SkyDome(city_model.dome_radius)
    multi_skydomes = MultiSkyDomes(skydome)
    
    suns = get_sun_and_sky(p, sunpath)    

    pp(suns)

    sun_analysis = SunAnalysis(city_model, city_results)
    sky_analysis = SkyAnalysis(city_model, city_results, skydome, multi_skydomes)
    
    #Execute analysis        
    if  (p.a_type == AnalysisType.sun_raycasting):    
        run_iterative_sun(sun_analysis, suns)
    elif(p.a_type == AnalysisType.sky_raycasting):    
        run_iterative_sky(sky_analysis, suns)
    elif(p.a_type == AnalysisType.com_raycasting):    
        run_combined(sun_analysis, sky_analysis, suns)
    elif(p.a_type == AnalysisType.sky_raycasting_some):    
        run_sky_domes(sky_analysis, suns)
        
    #Get geometry for the sunpath and current sun position
    if(p.prepare_display):
        viewer = Viewer()
        sunpath_mesh = SunpathMesh(city_model.sunpath_radius)
        color_mesh(p, city_results, viewer)
        sunpath_mesh.create_sunpath_diagram(suns, sunpath, city_model)                
        
        viewer.add_meshes(city_results.get_city_mesh_out())
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

    weather_file_clm = '../../../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'

    # Instant solar anaysis 
    args_1 = ['--inputfile', inputfile_L, 
              '--mode', '1',
              '--analysis', '1',
              '--one_date', '2019-03-30 09:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '2']

    # Instant sky analysis
    args_2 = ['--inputfile', inputfile_S, 
              '--mode', '1',
              '--analysis', '2',
              '--one_date', '2019-03-30 12:00:00', 
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '6']  
    
    # Instant combined analysis
    args_3 = ['--inputfile', inputfile_S, 
              '--mode', '1',
              '--analysis', '3',
              '--one_date', '2015-03-30 12:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm, 
              '--colorby', '3']

    # Iterative solar analysis 
    args_4 = ['--inputfile', inputfile_L, 
              '--mode', '2',
              '--analysis', '1',
              '--start_date', '2019-03-30 06:00:00',
              '--end_date', '2019-03-30 21:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm, 
              '--colorby', '3']

    # Iterative sky analysis 
    args_5 = ['--inputfile', inputfile_S, 
              '--mode', '2',
              '--analysis', '2',
              '--start_date', '2019-03-30 06:00:00',
              '--end_date', '2019-03-30 21:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '6']        
      
    # Iterative combined analysis
    args_6 = ['--inputfile', inputfile_S, 
              '--mode', '2',
              '--analysis', '3',
              '--start_date', '2019-03-30 06:00:00',
              '--end_date', '2019-03-30 21:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '3']   

    # Sky analysis with dome visualisation for debugging
    args_7 = ['--inputfile', inputfile_S, 
              '--mode', '1',
              '--analysis', '4',
              '--one_date', '2015-03-30 12:00:00',
              '--data_source', '3',
              '--w_file', weather_file_clm,
              '--colorby', '6']                

    run_script(args_1)


