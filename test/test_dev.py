import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)
import run_solar

inputfile_S = '/Users/jensolsson/Documents/Dev/DTCC/CitySolar/data/CitySurfaceS.stl'
inputfile_L = '/Users/jensolsson/Documents/Dev/DTCC/CitySolar/data/CitySurface69k.stl'

def test_for_dev():
    #assert run_solar.run(['--inputfile', inputfile_S, 
    #                       '-a', '1', 
    #                       '-tin', '6',
    #                       '--prep_disp', '1', 
    #                       '-disp', '1', 
    #                      '-c', '7'])
    pass
