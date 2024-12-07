# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:03:46 2024

@author: Bastien MOLCRETTE
"""
import numpy as np
import pandas as pd
import tkinter
from tkinter.filedialog import askopenfilenames
import re
import ntpath

def IcyXML2CSV():
    """
    Convert Icy-parsed XML trajectory files to CSV format TXY or TXYZ (T: Time, X-Y-Z: spatial coordinates)

    Just select the files you want to convert when asked; the CSV files are iteratively saved in the same location
    than original XML files with same basename.

    Parameters:
        No argument
    
    Returns:
        None
    """
    # Select the Icy-parsed XML files you want to convert to CSV
    root = tkinter.Tk(className='Open trajectories', )
    file_tracer = askopenfilenames()
    root.destroy()
    
    # Loop iterating over the list of selected files
    for j in range(len(file_tracer)):
        xml_data = open(file_tracer[j], 'r').read() # load XML file as text file
        res_x = [m.start() for m in re.finditer('x=', xml_data)] # create list of x-coordinates
        res_y = [m.start() for m in re.finditer('y=', xml_data)] # create list of y-coordinates
        res_z = [m.start() for m in re.finditer('z=', xml_data)] # create list of z-coordinates
        res_t = [m.start() for m in re.finditer('t=', xml_data)] # create list of time-coordinates
        res_type = [m.start() for m in re.finditer('type=', xml_data)] # list used to localize time coordinates
        list_X = []
        list_Y = []
        list_Z = []
        list_T = []
        for k in range(len(res_x)): # loop to extract the TXYZ coordinates over the trajectory
            try:
                list_X.append(float(xml_data[res_x[k]+3:res_y[k]-2])) # pick the x coordinate
            except ValueError:
                list_X.append(float('nan')) # if the coordinate is missing or corrupted (not a float value), replace it by a NaN
            try:
                list_Y.append(float(xml_data[res_y[k]+3:res_z[k]-2])) # pick the y coordinate
            except ValueError:
                list_Y.append(float('nan')) # if the coordinate is missing or corrupted (not a float value), replace it by a NaN
            try:
                list_Z.append(float(xml_data[res_z[k]+3:res_z[k]+3+xml_data[res_z[k]+3:res_z[k]+20].find('"')])) # pick the z coordinate
            except ValueError:
                list_Z.append(float('nan')) # if the coordinate is missing or corrupted (not a float value), replace it by a NaN
            try:
                list_T.append(int(float(xml_data[res_t[k]+3:res_type[k]-2]))) # pick the time coordinate
            except ValueError:
                list_T.append(float('nan')) # if the coordinate is missing or corrupted (not a float value), replace it by a NaN
        if False in (list_Z == np.zeros(len(res_x))): # if some z-coordinates are non-zeros (3D trajectory), add a fourth column to the CSV file for z-coordinates
            d_coord_list = {'t': list_T, 'x_tracer'+str(j): list_X, 'y_tracer'+str(j): list_Y, 'z_tracer'+str(j): list_Z}
        else: # for 2D trajectory, only three columns in CSV (TXY)
            d_coord_list = {'t': list_T, 'x_tracer'+str(j): list_X, 'y_tracer'+str(j): list_Y}
        df_xml = pd.DataFrame(data=d_coord_list) # create a pandas dataframe with coordinates for CSV conversion
        df_xml.to_csv(ntpath.basename(file_tracer[j])[:-4]+'.csv',header=False, index=False) # save coordinates into CSV format without header, no pandas indexes, with same basename as XML file and same folder
    print('Icy-XML conversion to CSV done!')
    return

if __name__ == "__main__":
   IcyXML2CSV()
else:
   print("Error occured while running script")