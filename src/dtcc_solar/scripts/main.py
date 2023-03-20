import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

import numpy as np
import pandas as pd
import time
import os
import trimesh
import argparse

from dtcc_solar import utils
from dtcc_solar import data_io 
from dtcc_solar.mysun import Sunpath
from dtcc_solar.solarviewer import SolarViewer
from dtcc_solar.utils import ColorBy, AnalysisType
from dtcc_solar.model import Model
from dtcc_solar.results import Results
from dtcc_solar.data_io import Parameters
from dtcc_solar.skydome import SkyDome
from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis
from dtcc_solar.combind_analysis import CombinedAnalysis
from dtcc_solar.multi_skydomes import MultiSkyDomes

def register_args(args):    
    
    default_path = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceL.stl'
    parser = argparse.ArgumentParser(description='Parameters to run city solar analysis', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--analysis'    , type=int  , metavar='', default=1, help=' sun_raycasting = 1, sky_raycasting = 2, sky_raycasting_some = 3, sun_raycast_iterative = 4')
    parser.add_argument('-lat', '--latitude'  , type=float, metavar='', default=51.5 , help='Latitude for location of analysis')
    parser.add_argument('-lon', '--longitude' , type=float, metavar='', default=-0.12, help='Longitude for location of analysis')
    parser.add_argument('-f', '--inputfile'   , type=str  , metavar='', default=default_path, help='Filename (incl. path) for city mesh (*.stl, *.vtu)')
    parser.add_argument('-d', '--one_date'    , type=str  , metavar='', default='2015-03-30 09:00:00', help='Date and time for instant analysis')
    parser.add_argument('-ds', '--start_date' , type=str  , metavar='', default='2015-03-30 07:00:00', help='Start date for iterative analysis')
    parser.add_argument('-de', '--end_date'   , type=str  , metavar='', default='2015-03-30 21:00:00', help='End date for iterative analysis')
    parser.add_argument('-disp', '--display'  , type=int  , metavar='', default=1, help='Display results with pyglet')
    parser.add_argument('-pd', '--prep_disp'  , type=int  , metavar='', default=1, help='Preproscess colors and other graphic for display')
    parser.add_argument('-o', '--origin'      , type=float, metavar='', default=[0,0,0], help='Origin for the model')
    parser.add_argument('-c', '--colorby'     , type=int  , metavar='', default=2, help='Colo_by: face_sun_angle =1, face_sun_angle_shadows = 2, face_irradiance = 3 , face_shadows = 4, face_in_sky = 5, face_diffusion = 6')
    parser.add_argument('-e', '--export'      , type=bool , metavar='', default=True, help='Export data')
    parser.add_argument('-ep', '--exportpath' , type=str  , metavar='', default='./data/dataExport.txt', help='Path for data export of type *.txt')
    new_args = parser.parse_args(args)
    return new_args

def print_args(args):
    for arg in vars(args):
        print(arg, '\t', getattr(args, arg))
    print("----------------------------------------")    
    
def run_instant(p:Parameters, sunpath:Sunpath, city_model:Model, sun_analysis:SunAnalysis, sky_analysis:SkyAnalysis, dict_keys, w_data):  
    #Get sun position for instant analysis
    sun_time = pd.to_datetime(p.one_date)
    sun_pos = sunpath.get_sun_position_for_a_date(sun_time, False, False) 
    sun_vec = utils.normalise_vector(city_model.origin - sun_pos[0])
    
    #Run analysis
    if(p.a_type == AnalysisType.sun_raycasting):
        sun_analysis.execute_raycasting(sun_vec)
        sun_analysis.set_city_mesh_out()

    elif(p.a_type == AnalysisType.sky_raycasting):
        sky_analysis.execute_raycasting(sun_vec, dict_keys, w_data)
        sky_analysis.set_city_mesh_out()

    elif(p.a_type == AnalysisType.sky_raycasting_some):
        sky_analysis.execute_raycasting_some(sun_vec)
        sky_analysis.set_city_mesh_out()
        sky_analysis.set_dome_mesh_out()
        
    return sun_pos

def run_iterative(p:Parameters, sunpath:Sunpath, city_model:Model, sun_analysis:SunAnalysis, dict_keys, dates, w_data):
    #Get multiple solar positions for iterative analysis     
    sun_positions = sunpath.get_sun_position_for_dates(dates, False, False)
    [sun_positions, dates, dict_keys] = sunpath.remove_position_under_horizon(city_model.horizon_z, sun_positions, dates, dict_keys)
    sun_vectors_dict = utils.get_sun_vecs_dict_from_sun_pos(sun_positions, city_model.origin, dict_keys)
    sun_analysis.execute_raycasting_iterative(sun_vectors_dict, dict_keys, w_data)
    sun_analysis.set_city_mesh_out()
    return sun_positions

def run_combined(sunpath:Sunpath, city_model:Model, com_analysis:CombinedAnalysis, dict_keys, dates, w_data):
    #Get multiple solar positions for iterative analysis      
    sun_positions = sunpath.get_sun_position_for_dates(dates, False, False)
    [sun_positions, dates, dict_keys] = sunpath.remove_position_under_horizon(city_model.horizon_z, sun_positions, dates, dict_keys)
    sun_vectors_dict = utils.get_sun_vecs_dict_from_sun_pos(sun_positions, city_model.origin, dict_keys)
    com_analysis.execute(sun_vectors_dict, dict_keys, w_data)
    com_analysis.set_city_mesh_out()
    return sun_positions

def export(p:Parameters, city_results:Results, exportpath):
    if p.color_by == ColorBy.face_sun_angle: 
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)
    elif p.color_by == ColorBy.face_sun_angle_shadows: 
        data_io.print_list(city_results.get_face_sun_angles(), exportpath)    
    elif p.color_by == ColorBy.face_irradiance: 
        data_io.print_list(city_results.get_face_irradiance(), exportpath)
    elif p.color_by == ColorBy.face_shadows: 
        data_io.print_list(city_results.get_face_in_sun(), exportpath)    

def color_mesh(p:Parameters, city_results:Results, viewer:SolarViewer):
    if(p.a_type == AnalysisType.sun_raycasting):
        city_results.color_city_mesh_from_sun(p.color_by)
    elif(p.a_type == AnalysisType.sky_raycasting):
        city_results.color_city_mesh_from_sky(p.color_by)
    elif(p.a_type == AnalysisType.sky_raycasting_some):
        city_results.color_dome_mesh(p.color_by)
        viewer.add_dome_mesh(city_results.get_dome_mesh_out())
    elif(p.a_type == AnalysisType.sun_raycast_iterative):
        city_results.color_city_mesh_iterative(p.color_by)
    elif(p.a_type == AnalysisType.com_iterative):
        city_results.color_city_mesh_com_iterative(p.color_by)    

def create_sunpath(sunpath:Sunpath, viewer:SolarViewer, city_model:Model, sun_positions, city_results:Results):
    [sunX, sunY, sunZ] = sunpath.get_sunpath_hour_loops(2019, 5, False, False)
    sun_path_meshes = viewer.create_sunpath_loops(sunX, sunY, sunZ, city_model.sunpath_radius)
    [sunX, sunY, sunZ] = sunpath.get_sunpath_day_loops(pd.to_datetime(['2015-06-21', '2015-03-21', '2015-12-21']), 10, False, False)
    sun_path_meshes_day = viewer.create_sunpath_loops(sunX, sunY, sunZ, city_model.sunpath_radius)
    sun_path_meshes.extend(sun_path_meshes_day)
    sun_mesh = viewer.create_solar_spheres(sun_positions, city_model.sun_size)
    viewer.add_meshes(city_results.get_city_mesh_out(), sun_mesh, sun_path_meshes)

###############################################################################################################################

def run_script(command_line_args):

    clock1 = time.perf_counter()
    os.system('clear')
    print("-------- Solar Analysis Started -------")

    args = register_args(command_line_args)
    print(command_line_args)

    #Convert command line input to enums and data formated for the analysis
    p = Parameters( args.analysis,  args.inputfile,  args.latitude,   args.longitude, 
                    args.prep_disp, args.display,    args.origin,     args.colorby,  
                    args.export,    args.one_date,   args.start_date, args.end_date)    
    
    print_args(args)

    #Main object for the analysis 
    [w_data, dict_keys, dates] = data_io.import_weather_data_clm(p)
    data_io.import_weather_date_epw(p)
    city_mesh = trimesh.load_mesh(p.file_name)
    city_model = Model(city_mesh)
    city_results = Results(city_model)
    skydome = SkyDome(city_model.dome_radius)
    multi_skydomes = MultiSkyDomes(skydome)
    sun_analysis = SunAnalysis(city_model, city_results)
    sky_analysis = SkyAnalysis(city_model, city_results, skydome, multi_skydomes)
    com_analysis = CombinedAnalysis(city_model, city_results, skydome, sun_analysis, sky_analysis)
    sunpath = Sunpath(p.latitude, p.longitude, city_model.sunpath_radius, p.origin)

    #Execute analysis        
    if(p.a_type == AnalysisType.sun_raycast_iterative):    
        sun_positions = run_iterative(p, sunpath, city_model, sun_analysis, dict_keys, dates, w_data)
    elif(p.a_type == AnalysisType.com_iterative):    
        sun_positions = run_combined(sunpath, city_model, com_analysis, dict_keys, dates, w_data)
    else:    
        sun_positions = run_instant(p, sunpath, city_model, sun_analysis, sky_analysis, dict_keys, w_data)
        
    #Get geometry for the sunpath and current sun position
    if(p.prepare_display):
        viewer = SolarViewer()
        color_mesh(p, city_results, viewer)
        create_sunpath(sunpath, viewer, city_model, sun_positions, city_results)                
        if(p.display):
            viewer.show()

    clock2 = time.perf_counter()
    print("Total computation time elapsed: " + str(round(clock2  - clock1,4)))
    print("----------------------------------------")
    return True                

if __name__ == "__main__":
    
    inputfile_S = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceS.stl'
    inputfile_M = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceM.stl'
    inputfile_L = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceL.stl'


    args_1 = ['--inputfile', inputfile_M, 
              '--analysis', '1',
              '--one_date', '2015-03-30 09:00:00',
              '--display', '1',
              '--colorby', '2']

    args_2 = ['--inputfile', inputfile_S, 
              '--analysis', '2',
              '--colorby', '6']

    args_3 = ['--inputfile', inputfile_S, 
              '--analysis', '3',
              '--one_date', '2015-03-30 12:00:00',
              '--colorby', '6']        

    args_4 = ['--inputfile', inputfile_L, 
              '--analysis', '4',
              '--one_date', '2015-03-30 09:00:00', 
              '--colorby', '3']

    args_5 = ['--inputfile', inputfile_S, 
              '--analysis', '5',
              '--start_date', '2015-03-30 06:00:00',
              '--end_date', '2015-03-30 21:00:00',
              '--colorby', '3']        

    args_6 = ['--inputfile', inputfile_L, 
              '--analysis', '5',
              '--start_date', '2015-03-30 07:00:00',
              '--end_date', '2015-03-30 21:00:00',
              '--colorby', '3']        


    run_script(args_1)


