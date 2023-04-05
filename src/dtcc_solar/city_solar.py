from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis

class CitySolar:

    def __init__(self, model, results):
        self.sun_analysis = SunAnalysis(model, results)
        self.sky_analysis = SkyAnalysis(model, results)

    def sun_raycasting(self, sun_vec):     
        self.sun_analysis.execute_raycasting(sun_vec)

    def sun_raycasting_iterative(self, sunVecs): 
        self.sun_analysis.execute_raycasting_iterative(sunVecs)

    def sky_raycasting(self, sun_vec):
        self.sky_analysis.create_skydome()
        self.sky_analysis.execute_raycasting()
        
    def sky_raycasting_some(self, sun_vec):
        self.sky_analysis.create_skydome()    
        self.sky_analysis.execute_raycasting_some(sun_vec)    
    