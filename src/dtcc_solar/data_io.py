import numpy as np
import pandas as pd
import csv
import math
import typing

from csv import reader
from dtcc_solar.utils import Vec3
from typing import Dict, List
from pprint import pp
from dataclasses import dataclass
from shapely import LinearRing, Point

        
def export_results(solpos):
    with open("sunpath.txt", "w") as f:
        
        for item in solpos['zenith'].values:
            f.write(str(item[0]) + '\n')

def print_list(listToPrint, path):
    counter = 0
    with open(path, 'w') as f:
        for row in listToPrint:
            f.write( str(row) + '\n')
            counter += 1 

    print("Export completed")

def print_dict(dictToPrint, filename):
    counter = 0
    with open(filename, "w") as f:
        for key in dictToPrint:
            f.write('Key:' + str(key) + ' ' + str(dictToPrint[key])+'\n')

def print_results(shouldPrint,faceRayFaces):
    counter = 0
    if shouldPrint:
        with open("faceRayFace.txt", "w") as f:
            for key in faceRayFaces:
                f.write('Face index:' + str(key) + ' ' + str(faceRayFaces[key])+'\n')
                counter += 1 

    print(counter)

def write_to_csv(filename:str, data):

    print("Write to CSV")
    print(type(data))

    data_list = data.to_list()

    print(type(data_list))

    with open(filename, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        for d in data:
            filewriter.writerow(str(d))
        
def read_sunpath_diagram_from_csv_file(filename):

    pts = []
    with open(filename, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                pt = Vec3(x = float(row[0]), y = float(row[1]), z = float(row[2]))                
                pts.append(pt)

    return pts

def match_sunpath_scale(loop_pts:Dict[int, List[Vec3]], radius: float) -> Dict[int, List[Vec3]]:
    # Calculate the correct scale factor for the imported sunpath diagram
    pt = loop_pts[0][0]
    current_raduis = math.sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z) 
    sf = radius / current_raduis

    for hour in loop_pts:
        print(len(loop_pts[hour]))
        for i in range(len(loop_pts[hour])):
            pt = loop_pts[hour][i]
            pt.x = sf * pt.x 
            pt.y = sf * pt.y
            pt.z = sf * pt.z
            loop_pts[hour][i] = pt

    return loop_pts

