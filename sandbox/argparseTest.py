import argparse
import os 

parser = argparse.ArgumentParser(description='Parameters to run city solar')
parser.add_argument('-a', '--analysis'   , type=int, metavar='', default=1 , help='Instant analysis = 1, iterative analysis = 2')
parser.add_argument('-lat', '--latitude' , type=float, metavar='', default=51.5 , help='Latitude for location of analysis')
parser.add_argument('-lon', '--longitude', type=float, metavar='', default=-0.12, help='Longitude for location of analysis')
parser.add_argument('-f', '--file_name'   , type=str, metavar='', default='data/CitySurface69k.stl', help='Filename (incl. path) for city mesh')
parser.add_argument('-d', '--one_date'       , type=str, metavar='', default='2019-03-30 09:00:00', help='Date and time for solar position')
parser.add_argument('-ds', '--start_date' , type=str, metavar='', default='2019-03-30 09:00:00', help='Start date for iterative analysis')
parser.add_argument('-de', '--end_date'   , type=str, metavar='', default='2019-03-30 22:00:00', help='End date for iterative analysis')
parser.add_argument('-ss', '--step_size' , type=str, metavar='', default='1H', help='Step size for iterative analysis ("0.5H", "1H", "5D",...)')
parser.add_argument('-disp', '--display' , type=bool, metavar='', default='1H', help='Step size for iterative analysis ("0.5H", "1H", "5D",...)')
args = parser.parse_args()

if __name__ == '__main__':
    
    os.system('clear')
    
    for arg in vars(args):
        print(arg, '\t', getattr(args, arg))

    #print(args)


    #result_data = hello_world(lang=args.lang, sleep_time=args.sleep_time)
    
