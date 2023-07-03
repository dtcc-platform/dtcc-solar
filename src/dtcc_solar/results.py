
import numpy as np
import dtcc_solar.mesh_compute as mc
from dtcc_solar.utils import ColorBy, Sun, Output, OutputAcum
from typing import List, Dict


#This class contains all the results from analysis that will be accessed for visualisation
class Results:

    res_list:List[Output]
    res_acum:OutputAcum
    res_avrg:OutputAcum

    def __init__(self, suns:List[Sun], face_count:int):
        self.res_list = self.create_res_list(suns, face_count)
        self.res_acum = self.create_acum_res(suns, face_count)
        self.res_avrg = self.create_acum_res(suns, face_count)

    def create_res_list(self, sun_dates:List[Sun], face_count:int):
        results = []
        empty_float_array = np.zeros(face_count, dtype= float)
        empty_bool_array = np.zeros(face_count, dtype= bool)
        counter = 0
        for sun in sun_dates:
            date_str = sun.datetime_str
            date_ts = sun.datetime_ts

            res = Output(datetime_str = date_str,
                    datetime_ts = date_ts,
                    index = counter, 
                    face_sun_angles = empty_float_array.copy(),
                    face_in_sun = empty_bool_array.copy(),
                    face_in_sky = empty_bool_array.copy(),
                    face_irradiance_dh = empty_float_array.copy(),
                    face_irradiance_dn = empty_float_array.copy(),
                    face_irradiance_di = empty_float_array.copy(),
                    face_irradiance_tot = empty_float_array.copy())

            counter += 1
            results.append(res)
            
        return results  
    
    def create_acum_res(self, suns_dates:List[Sun], face_count:int):
        
        start_date = suns_dates[0].datetime_ts
        end_date = suns_dates[-1].datetime_ts
        empty_float_array = np.zeros(face_count, dtype= float)
        empty_bool_array = np.zeros(face_count, dtype= bool)

        res_acum = OutputAcum( start_datetime_str = str(start_date),
                            start_datetime_ts = start_date, 
                            end_datetime_str= str(end_date), 
                            end_datetime_ts = end_date,
                            face_sun_angles = empty_float_array.copy(),
                            face_in_sun = empty_bool_array.copy(),
                            face_in_sky = empty_bool_array.copy(),
                            face_irradiance_dh = empty_float_array.copy(),
                            face_irradiance_dn = empty_float_array.copy(),
                            face_irradiance_di = empty_float_array.copy(),
                            face_irradiance_tot = empty_float_array.copy())

        return res_acum

    def calc_accumulated_results(self):
        for res in self.res_list:
            self.res_acum.face_sun_angles += res.face_sun_angles
            self.res_acum.face_in_sun += res.face_in_sun
            self.res_acum.face_irradiance_dh += res.face_irradiance_dh
            self.res_acum.face_irradiance_dn += res.face_irradiance_dn
            self.res_acum.face_irradiance_di += res.face_irradiance_di
            self.res_acum.face_irradiance_tot += res.face_irradiance_dh + res.face_irradiance_dn + res.face_irradiance_di 

    def calc_average_results(self):
        n = len(self.res_list)
        for res in self.res_list:
            self.res_avrg.face_sun_angles += res.face_sun_angles / n
            self.res_avrg.face_irradiance_dh += (res.face_irradiance_dh / n)
            self.res_avrg.face_irradiance_dn += (res.face_irradiance_dn / n)
            self.res_avrg.face_irradiance_di += (res.face_irradiance_di / n)
        
    
    ###################### Getters an setters for city mesh results #######################

 

