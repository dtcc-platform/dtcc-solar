import numpy as np
import pandas as pd
from dtcc_solar.scripts.main import run_script


class TestSolar:

    inputfile_S: str
    inputfile_M: str
    inputfile_L: str

    weather_file_clm:str
    weather_file_epw:str
    
    def setup_method(self):    
        self.inputfile_S = '../data/models/CitySurfaceS.stl'
        self.inputfile_M = '../data/models/CitySurfaceM.stl'
        self.inputfile_L = '../data/models/CitySurfaceL.stl'

        self.weather_file_clm = '../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'
        self.weather_file_epw = '../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.epw'
        

########################### Testig of instant analysis ##############################
    def test_instant_face_sun_angle(self):
        assert run_script([  '--inputfile', self.inputfile_L, 
                            '--analysis', '1', 
                            '--prep_disp', '1', 
                            '--display', '0', 
                            '--data_source', '3',
                            '--w_file', self.weather_file_clm,
                            '--colorby', '1']) 

    def test_instant_face_sun_angle_shadows(self):
        assert run_script([  '--inputfile', self.inputfile_L, 
                            '--analysis', '1',
                            '--prep_disp', '1', 
                            '--display', '0',
                            '--data_source', '3',
                            '--w_file', self.weather_file_clm,
                            '--colorby', '2']) 

    def test_instant_face_irradiance(self):
        assert run_script([  '--inputfile', self.inputfile_L,
                            '--analysis', '1',
                            '--prep_disp', '1', 
                            '--display', '0',
                            '--data_source', '3', 
                            '--w_file', self.weather_file_clm,
                            '--colorby', '3']) 
    
    def test_instant_face_shadows(self):    
        assert run_script([  '--inputfile', self.inputfile_L, 
                            '--analysis', '1',
                            '--prep_disp', '1',
                            '--display', '0',
                            '--data_source', '3',
                            '--w_file', self.weather_file_clm,
                            '--colorby', '4'])

########################################################################################


########################### Testing diffusion calculations ##############################
    def test_sky_raycasting(self):    
        assert run_script([  '--inputfile', self.inputfile_S,
                            '--analysis', '2',
                            '--prep_disp', '1',
                            '--display', '0',
                            '--data_source', '3',
                            '--w_file', self.weather_file_clm,
                            '--colorby', '6'])

    def test_sky_raycasting_some(self):    
        assert run_script([  '--inputfile', self.inputfile_S,
                            '--analysis', '3',
                            '--prep_disp', '1',
                            '--display', '0',
                            '--data_source', '3',
                            '--w_file', self.weather_file_clm,
                            '--colorby', '6'])

    #Iterative testing
    def test_iterative_irradiance(self):
        assert run_script([  '--inputfile', self.inputfile_M, 
                            '--analysis', '4', 
                            '--prep_disp', '1', 
                            '--display', '0',
                            '--data_source', '3',
                            '--w_file', self.weather_file_clm,
                            '--colorby', '3'])
    