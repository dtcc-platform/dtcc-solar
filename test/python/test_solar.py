import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

import numpy as np
import pandas as pd
from dtcc_solar.scripts.run_solar import run

inputfile_S = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceS.stl'
inputfile_M = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceS.stl'
inputfile_L = '/Users/jensolsson/Documents/Dev/DTCC/dtcc-solar/data/models/CitySurfaceL.stl'


########################### Testig of instant analysis ##############################
def test_instant_face_sun_angle():
    assert run([  '--inputfile', inputfile_L, 
                            '--analysis', '1', 
                            '--prep_disp', '1', 
                            '--display', '0', 
                            '--colorby', '1']) 

def test_instant_face_sun_angle_shadows():
    assert run([  '--inputfile', inputfile_L, 
                            '--analysis', '1',
                            '--prep_disp', '1', 
                            '--display', '0',
                            '--colorby', '2']) 

def test_instant_face_irradiance():
    assert run([  '--inputfile', inputfile_L,
                            '--analysis', '1',
                            '--prep_disp', '1', 
                            '--display', '0', 
                            '--colorby', '3']) 
    
def test_instant_face_shadows():    
    assert run([  '--inputfile', inputfile_L, 
                            '--analysis', '1',
                            '--prep_disp', '1',
                            '--display', '0',
                            '--colorby', '4'])

########################################################################################


########################### Testing diffusion calculations ##############################
def test_sky_raycasting():    
    assert run([  '--inputfile', inputfile_S,
                            '--analysis', '2',
                            '--prep_disp', '1',
                            '--display', '0',
                            '--colorby', '6'])

def test_sky_raycasting_some():    
    assert run([  '--inputfile', inputfile_S,
                            '--analysis', '3',
                            '--prep_disp', '1',
                            '--display', '0',
                            '--colorby', '6'])


#Iterative testing
def test_iterative_irradiance():
    assert run([  '--inputfile', inputfile_L, 
                            '--analysis', '4', 
                            '--prep_disp', '1', 
                            '--display', '0',
                            '--colorby', '3'])
    