# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:32:31 2023

@author: molcretb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
import tkinter
from tkinter.filedialog import asksaveasfile, askopenfilenames
import re
import ntpath
import os.path
from matplotlib import colors
import time
import json


#calib_px = 0.11 #1 px = 0.11 µm
#calib_fr = 0.5 #1 frame = 0.5 s




# def load_csv_traj_orig(file_tracer):
#     """
#     Load CSV trajectory files in TXY or TXYZ format (T: Time, X-Y-Z: spatial coordinates)

#     Parameters:
#         List of CSV file paths (str variables) containing the trajectories to be analyzed
    
#     Returns:
#         List of dataframes each containing the coordinates of a single curated trajectory
#     """
    
#     list_df = {} # create an empty list that will contain dataframes for each single trajectory
#     for j in range(len(file_tracer)): # loop over each trajectory of the list
#         df_csv = pd.read_csv(file_tracer[j], header=None) # load the trajectory from the related CSV file without the header (if there is a header) and create a dataframe with coordinates
#         orig_size = df_csv[0].size # original length of the trajectory, used to detect missing/corrupted coordinates and tell user about it
#         df_csv = df_csv.apply(pd.to_numeric, errors='coerce') # used to convert str values to float and convert non-numerical values to Nan
#         df_csv= df_csv.dropna() # remove frames with missing/corrupted coordinates
#         df_csv = df_csv.drop(df_csv[df_csv[0]<0].index) # remove frames with negative time values
#         df_csv = df_csv.drop(df_csv[df_csv[0].duplicated()].index) # remove duplicates of frames (if two or more frames share the same time value)
#         list_df[j] = df_csv # copy the current dataframe (e.g. single curated trajectory) into the list used as output
#         NanCurr_size = df_csv[0].size # length of the curated trajectory
#         if NanCurr_size < orig_size: #if some frames have been removed (missing/corrupted coordinates), inform the user about it
#             print(ntpath.basename(file_tracer[j]) + ': ' + str(orig_size-NanCurr_size) + ' missing/errors/negative frames have been curated')
#     return list_df # return the list of curated trajectories as dataframes

def load_csv_traj(file_tracer):
    """
    Load CSV trajectory files in TXY or TXYZ format (T: Time, X-Y-Z: spatial coordinates)

    Parameters:
        List of CSV file paths (str variables) containing the trajectories to be analyzed
    
    Returns:
        List of dataframes each containing the coordinates of a single curated trajectory
    """
    df_csv = pd.read_csv(file_tracer, header=None) # load the trajectory from the related CSV file without the header (if there is a header) and create a dataframe with coordinates
    orig_size = df_csv[0].size # original length of the trajectory, used to detect missing/corrupted coordinates and tell user about it
    df_csv = df_csv.apply(pd.to_numeric, errors='coerce') # used to convert str values to float and convert non-numerical values to Nan
    df_csv= df_csv.dropna() # remove frames with missing/corrupted coordinates
    df_csv = df_csv.drop(df_csv[df_csv[0]<0].index) # remove frames with negative time values
    df_csv = df_csv.drop(df_csv[df_csv[0].duplicated()].index) # remove duplicates of frames (if two or more frames share the same time value)
    NanCurr_size = df_csv[0].size # length of the curated trajectory
    if NanCurr_size < orig_size: #if some frames have been removed (missing/corrupted coordinates), inform the user about it
        print(ntpath.basename(file_tracer) + ': ' + str(orig_size-NanCurr_size) + ' missing/errors/negative frames have been curated')
    return df_csv # return the list of curated trajectories as dataframes


# def load_csv_traj_couple(file_tracer): # need extra work, maybe merging the two load_csv as they are really similar
#     """
#     Load coupled CSV trajectory files in TXY or TXYZ format (T: Time, X-Y-Z: spatial coordinates), which need to be analyzed by pairs

#     Parameters:
#         Path (str variable) to a CSV file containing a list of trajectory file names that need to be analyzed by pairs: traj1 (column 1), traj2 (column 2)
    
#     Returns:
#         List of coupled trajectories dataframes each containing the coordinates of a single curated trajectory
#     """
#     df_csv = pd.read_csv(file_tracer, header=None)
#     orig_size = df_csv[0].size
#     df_csv = df_csv.apply(pd.to_numeric, errors='coerce')
#     df_csv= df_csv.dropna()
#     df_csv = df_csv.drop(df_csv[df_csv[0]<0].index)
#     list_df = df_csv.drop(df_csv[df_csv[0].duplicated()].index)
#     NanCurr_size = list_df[0].size
#     if NanCurr_size < orig_size:
#         print(ntpath.basename(file_tracer) + ': ' + str(orig_size-NanCurr_size) + ' missing/errors/negative frames have been curated')
#     return list_df


# def load_xml_traj(file_tracer):
#     list_df = {}
#     for j in range(len(file_tracer)):
#         xml_data = open(file_tracer[j], 'r').read()
#         res_x = [m.start() for m in re.finditer('x=', xml_data)]
#         res_y = [m.start() for m in re.finditer('y=', xml_data)]
#         res_z = [m.start() for m in re.finditer('z=', xml_data)]
#         res_t = [m.start() for m in re.finditer('t=', xml_data)]
#         res_type = [m.start() for m in re.finditer('type=', xml_data)]
#         list_X = []
#         list_Y = []
#         list_Z = []
#         list_T = []
#         for k in range(len(res_x)):
#             try:
#                 list_X.append(float(xml_data[res_x[k]+3:res_y[k]-2]))
#             except ValueError:
#                 list_X.append(float('nan'))
#             try:
#                 list_Y.append(float(xml_data[res_y[k]+3:res_z[k]-2]))
#             except ValueError:
#                 list_Y.append(float('nan'))
#             try:
#                 list_Z.append(float(xml_data[res_z[k]+3:res_z[k]+3+xml_data[res_z[k]+3:res_z[k]+20].find('"')]))
#             except ValueError:
#                 list_Z.append(float('nan'))
#             try:
#                 list_T.append(int(float(xml_data[res_t[k]+3:res_type[k]-2])))
#             except ValueError:
#                 list_T.append(float('nan'))
#         if False in (list_Z == np.zeros(len(res_x))):
#             d_coord_list = {'t': list_T, 'x_tracer'+str(j): list_X, 'y_tracer'+str(j): list_Y, 'z_tracer'+str(j): list_Z}
#         else:
#             d_coord_list = {'t': list_T, 'x_tracer'+str(j): list_X, 'y_tracer'+str(j): list_Y}
#         df_xml = pd.DataFrame(data=d_coord_list)
#         orig_size = df_xml['t'].size
#         df_xml = df_xml.apply(pd.to_numeric, errors='coerce')
#         df_xml = df_xml.dropna()
#         df_xml = df_xml.drop(df_xml[df_xml['t']<0].index)
#         list_df[j] = df_xml.drop(df_xml[df_xml['t'].duplicated()].index)
#         NanCurr_size = list_df[j]['t'].size
#         if NanCurr_size < orig_size:
#             print(ntpath.basename(file_tracer[j]) + ': ' + str(orig_size-NanCurr_size) + ' missing/errors/negative frames have been curated')
#     return list_df

def load_xml_traj(file_tracer): # original is xml_traj_couple
    xml_data = open(file_tracer, 'r').read()
    res_x = [m.start() for m in re.finditer('x=', xml_data)]
    res_y = [m.start() for m in re.finditer('y=', xml_data)]
    res_z = [m.start() for m in re.finditer('z=', xml_data)]
    res_t = [m.start() for m in re.finditer('t=', xml_data)]
    res_type = [m.start() for m in re.finditer('type=', xml_data)]
    list_X = []
    list_Y = []
    list_Z = []
    list_T = []
    for k in range(len(res_x)):
        try:
            list_X.append(float(xml_data[res_x[k]+3:res_y[k]-2]))
        except ValueError:
            list_X.append(float('nan'))
        try:
            list_Y.append(float(xml_data[res_y[k]+3:res_z[k]-2]))
        except ValueError:
            list_Y.append(float('nan'))
        try:
            list_Z.append(float(xml_data[res_z[k]+3:res_z[k]+3+xml_data[res_z[k]+3:res_z[k]+20].find('"')]))
        except ValueError:
            list_Z.append(float('nan'))
        try:
            list_T.append(int(float(xml_data[res_t[k]+3:res_type[k]-2])))
        except ValueError:
            list_T.append(float('nan'))
    if False in (list_Z == np.zeros(len(res_x))):
        d_coord_list = {'t': list_T, 'x_tracer': list_X, 'y_tracer': list_Y, 'z_tracer': list_Z}
    else:
        d_coord_list = {'t': list_T, 'x_tracer': list_X, 'y_tracer': list_Y}
    list_df = pd.DataFrame(data=d_coord_list)
    orig_size = list_df['t'].size
    list_df = list_df.apply(pd.to_numeric, errors='coerce')
    list_df = list_df.dropna()
    list_df = list_df.drop(list_df[list_df['t']<0].index)
    list_df = list_df.drop(list_df[list_df['t'].duplicated()].index)
    NanCurr_size = list_df['t'].size
    if NanCurr_size < orig_size:
        print(ntpath.basename(file_tracer) + ': ' + str(orig_size-NanCurr_size) + ' missing/errors/negative frames have been curated')
    return list_df

def gpytool(calib_px=1, calib_fr=1, calib_px_z = 1):
    root = tkinter.Tk(className='Open trajectories', )
    file_tracer = askopenfilenames()
    if file_tracer[0][-3:] == 'xml':
        df_traj = {}
        for j in range(len(file_tracer)):
            df_traj[j] = load_xml_traj(file_tracer[j])
    elif file_tracer[0][-3:] == 'csv':
        df_traj = {}
        for j in range(len(file_tracer)):
            df_traj[j] = load_csv_traj(file_tracer[j])
    else:
        raise ValueError("Only xml or csv files!")
    file_save = asksaveasfile(title="Save results", initialfile=ntpath.basename(file_tracer[0][0:len(file_tracer[0])-4])+'_all_list', defaultextension=".csv", filetypes=(("csv file", "*.csv"),))
    file_save.close()
    root.destroy()
    df_gpytool = pd.DataFrame(data={'traj_ID': [], 'alpha':[], 'D (um2/s**a)':[], 'traj_length':[]})
    for i in range(len(df_traj)):
        print(str(i+1)+' / '+str(len(df_traj)))
        if df_traj[0].shape[1] == 3:
            traj_exp = np.array([(df_traj[i][df_traj[i].keys()[1]]*calib_px).tolist(), (df_traj[i][df_traj[i].keys()[2]]*calib_px).tolist()]).T
        else:
            traj_exp = np.array([(df_traj[i][df_traj[i].keys()[1]]*calib_px).tolist(), (df_traj[i][df_traj[i].keys()[2]]*calib_px).tolist(), (df_traj[i][df_traj[i].keys()[3]]*calib_px_z).tolist()]).T
        vec_t = np.array([(df_traj[i][df_traj[i].keys()[0]]*calib_fr).tolist()]).T
        res = get_D_alpha(traj_exp-np.mean(traj_exp,axis=0),vec_t)
        #res.x[0] = res.x[0]*calib_px**2*calib_fr**(-res.x[1])
        df_gpytool = pd.concat([df_gpytool, pd.DataFrame(data={'traj_ID': [file_tracer[i]], 'alpha':[res.x[1]], 'D (um2/s**a)':[res.x[0]], 'traj_length':[len(vec_t)]})])
    df_gpytool.to_csv(file_save.name, index=False)
    return df_gpytool

def gpytool_couple(calib_px=1, calib_fr=1,calib_px_z=1):
    root = tkinter.Tk(className='Open trajectories', )
    file_traj_couple = askopenfilenames()
    root.destroy()
    list_traj_couple = pd.read_csv(file_traj_couple[0])
    path_folder = os.path.dirname(file_traj_couple[0])
    dict_json = {}
    savefile_ID = str(int(time.time()))
    df_gpytool = pd.DataFrame(data={'traj_ID': [], 'alpha':[], 'D (um2/s**a)':[], 'traj_length':[]}) #used for storage of individual corrected traj metadata
    for i in range(len(list_traj_couple)):
        print(str(i+1)+'/'+str(len(list_traj_couple)))
        if list_traj_couple['traj_1'][0][-3:] == 'xml':
            traj1 = load_xml_traj(path_folder+'/'+list_traj_couple['traj_1'][i])
            traj2 = load_xml_traj(path_folder+'/'+list_traj_couple['traj_2'][i])
        elif list_traj_couple['traj_1'][0][-3:] == 'csv':
            traj1 = load_csv_traj(path_folder+'/'+list_traj_couple['traj_1'][i])
            traj2 = load_csv_traj(path_folder+'/'+list_traj_couple['traj_2'][i])
        else:
            raise ValueError("Only xml or csv files!")
        traj1_common = traj1[traj1[traj1.keys()[0]].isin(traj2[traj2.keys()[0]])]
        traj2_common = traj2[traj2[traj2.keys()[0]].isin(traj1[traj1.keys()[0]])]
        if traj1_common.shape[1] == 3:
            traj_exp1 = np.array([(traj1_common[traj1_common.keys()[1]]*calib_px).tolist(), (traj1_common[traj1_common.keys()[2]]*calib_px).tolist()]).T
            traj_exp1 = traj_exp1-traj_exp1[0,:]
            traj_exp2 = np.array([(traj2_common[traj2_common.keys()[1]]*calib_px).tolist(), (traj2_common[traj2_common.keys()[2]]*calib_px).tolist()]).T
            traj_exp2 = traj_exp2-traj_exp2[0,:]
        else:
            traj_exp1 = np.array([(traj1_common[traj1_common.keys()[1]]*calib_px).tolist(), (traj1_common[traj1_common.keys()[2]]*calib_px).tolist(), (traj1_common[traj1_common.keys()[3]]*calib_px_z).tolist()]).T
            traj_exp1 = traj_exp1-traj_exp1[0,:]
            traj_exp2 = np.array([(traj2_common[traj2_common.keys()[1]]*calib_px).tolist(), (traj2_common[traj2_common.keys()[2]]*calib_px).tolist(), (traj2_common[traj2_common.keys()[3]]*calib_px_z).tolist()]).T
            traj_exp2 = traj_exp2-traj_exp2[0,:]
        vec_t = np.array([(traj1_common[traj1_common.keys()[0]]*calib_fr).tolist()]).T
        [res, traj_back_exp] = get_D_alpha_couple(traj_exp1, traj_exp2, vec_t)
        traj_back_exp = np.concatenate((vec_t, traj_back_exp), axis = 1)
        #res.x[0] = res.x[0]*calib_px**2*calib_fr**(-res.x[1])
        #res.x[2] = res.x[2]*calib_px**2*calib_fr**(-res.x[3])
        #res.x[4] = res.x[4]*calib_px**2*calib_fr**(-res.x[5])
        df_gpytool_couple = {'traj1': [list_traj_couple['traj_1'][i]],'traj2': [list_traj_couple['traj_2'][i]], 'alpha_1':[res.x[1]], 'D1 (um2/s**a)':[res.x[0]],'alpha_2':[res.x[3]], 'D2 (um2/s**a)':[res.x[2]],'alpha_back':[res.x[5]], 'D_back (um2/s**a)':[res.x[4]], 'traj_length':[len(vec_t)], 'traj_back_exp':traj_back_exp.T.tolist()}
        dict_json[i] = df_gpytool_couple
        Path(path_folder+'/output/').mkdir(parents=True, exist_ok=True)
        with open(path_folder+'/output/'+'results_'+savefile_ID+'.json', "w") as outfile:
            json.dump(dict_json, outfile)
        Path(path_folder+'/substrate_corrected/').mkdir(parents=True, exist_ok=True)
        if traj1_common.shape[1] == 3:
            traj1_corrected = pd.DataFrame(data={'t':vec_t[:,0], 'x':traj_exp1[:,0]-traj_back_exp[:,1], 'y':traj_exp1[:,1]-traj_back_exp[:,2]})
            traj2_corrected = pd.DataFrame(data={'t':vec_t[:,0], 'x':traj_exp2[:,0]-traj_back_exp[:,1], 'y':traj_exp2[:,1]-traj_back_exp[:,2]})
        else:
            traj1_corrected = pd.DataFrame(data={'t':vec_t[:,0], 'x':traj_exp1[:,0]-traj_back_exp[:,1], 'y':traj_exp1[:,1]-traj_back_exp[:,2], 'z':traj_exp1[:,2]-traj_back_exp[:,3]})
            traj2_corrected = pd.DataFrame(data={'t':vec_t[:,0], 'x':traj_exp2[:,0]-traj_back_exp[:,1], 'y':traj_exp2[:,1]-traj_back_exp[:,2], 'z':traj_exp2[:,2]-traj_back_exp[:,3]})
        traj1_corrected.to_csv(path_folder+'/substrate_corrected/'+list_traj_couple['traj_1'][i][:-4]+'_subs_corrected.csv', index=False)
        traj2_corrected.to_csv(path_folder+'/substrate_corrected/'+list_traj_couple['traj_2'][i][:-4]+'_subs_corrected.csv', index=False)
        df_gpytool = pd.concat([df_gpytool, pd.DataFrame(data={'traj_ID': [path_folder+'/substrate_corrected/'+list_traj_couple['traj_1'][i][:-4]+'_subs_corrected.csv'], 'alpha':[res.x[1]], 'D (um2/s**a)':[res.x[0]], 'traj_length':[len(vec_t)]})])
        df_gpytool = pd.concat([df_gpytool, pd.DataFrame(data={'traj_ID': [path_folder+'/substrate_corrected/'+list_traj_couple['traj_2'][i][:-4]+'_subs_corrected.csv'], 'alpha':[res.x[3]], 'D (um2/s**a)':[res.x[2]], 'traj_length':[len(vec_t)]})])
    df_gpytool.to_csv(path_folder+'/output/'+'results_'+savefile_ID+'.csv', index=False)
    return dict_json

def calcul_fbm_kernel(D, alpha, vec_t):
    mat_t, yv = np.meshgrid(vec_t, vec_t)
    fbm_kernel = D*(mat_t**alpha+mat_t.T**alpha-np.abs(mat_t-mat_t.T)**alpha)
    return fbm_kernel

def negLogPost(x, r, mu, vec_t):
    neg_log_Post = 0.5*np.matmul(np.transpose(r-mu),np.matmul(np.linalg.inv(calcul_fbm_kernel(x[0],x[1],vec_t)),r-mu)) \
    + 0.5*np.linalg.slogdet(calcul_fbm_kernel(x[0],x[1],vec_t))[1] + len(vec_t)/2*np.log(2*np.pi)
    return neg_log_Post.diagonal().prod()

def negLogPost_couple(x, r1, r2, vec_t):
    r_couple = np.concatenate((r1,r2))
    x = np.abs(x)
    A = calcul_fbm_kernel(x[0],x[1],vec_t)+calcul_fbm_kernel(x[4],x[5],vec_t)
    B = calcul_fbm_kernel(x[4],x[5],vec_t)
    C = calcul_fbm_kernel(x[2],x[3],vec_t)+calcul_fbm_kernel(x[4],x[5],vec_t)
    cov1 = np.concatenate((A, B),axis=1)
    cov2 = np.concatenate((B ,C),axis=1)
    cov_couple = np.concatenate((cov1,cov2))
    neg_log_Post = 0.5*np.matmul(np.transpose(r_couple),np.matmul(np.linalg.inv(cov_couple),r_couple)) \
    + 0.5*np.linalg.slogdet(cov_couple)[1] + len(vec_t)/2*np.log(2*np.pi)
    return neg_log_Post.diagonal().prod()

def generate_2Dtraj(N,D,alpha):
    trajectory = np.zeros([N, 2])
    trajectory[:,0] = np.random.multivariate_normal(np.zeros(N), calcul_fbm_kernel(D, alpha, np.linspace(1,N,N)))
    trajectory[:,1] = np.random.multivariate_normal(np.zeros(N), calcul_fbm_kernel(D, alpha, np.linspace(1,N,N)))
    return trajectory

def generate_3Dtraj(N,D,alpha):
    trajectory = np.zeros([N, 3])
    trajectory[:,0] = np.random.multivariate_normal(np.zeros(N), calcul_fbm_kernel(D, alpha, np.linspace(1,N,N)))
    trajectory[:,1] = np.random.multivariate_normal(np.zeros(N), calcul_fbm_kernel(D, alpha, np.linspace(1,N,N)))
    trajectory[:,2] = np.random.multivariate_normal(np.zeros(N), calcul_fbm_kernel(D, alpha, np.linspace(1,N,N)))
    return trajectory

def get_D_alpha(traj, *args):
    if len(args) == 0:
        vec_t = np.linspace(1,len(traj),len(traj))
    else:
        vec_t = args[0] + 1
    scale_factor = np.mean(np.abs(np.diff(traj,axis=0)))
    traj = traj / scale_factor
    mu = np.zeros(traj.shape)
    #mu[:,0] = np.mean(traj[:,0])
    #mu[:,1] = np.mean(traj[:,1])
    res = minimize(negLogPost, [np.mean(np.abs(np.diff(traj,axis=0)))**2, 1], method='Nelder-Mead', args=(traj, mu, vec_t))
    res.x[0] = res.x[0]*scale_factor**2
    return res

def get_D_alpha_couple(traj1, traj2, *args):
    if len(args) == 0:
        vec_t = np.linspace(1,len(traj1),len(traj1))
    else:
        vec_t = args[0] + 1
    scale_factor = np.mean([np.mean(np.abs(np.diff(traj1,axis=0))), np.mean(np.abs(np.diff(traj2,axis=0)))])
    traj1 = traj1 / scale_factor
    traj2 = traj2 / scale_factor
    estimate_alpha = get_D_alpha(traj1-traj2)
    estimate_subs = get_D_alpha(traj1)
    res = minimize(negLogPost_couple, [np.mean(np.abs(np.diff(traj1,axis=0)))**2, estimate_alpha.x[1], np.mean(np.abs(np.diff(traj2,axis=0)))**2, estimate_alpha.x[1], estimate_subs.x[0], estimate_subs.x[1]], method='Nelder-Mead', args=(traj1, traj2, vec_t),options={'maxfev': 5000})
    res.x = np.abs(res.x)
    res.x[0] = res.x[0]*scale_factor**2
    res.x[2] = res.x[2]*scale_factor**2
    res.x[4] = res.x[4]*scale_factor**2 
    cov1 = np.linalg.inv(calcul_fbm_kernel(res.x[0], res.x[1], vec_t))
    cov2 = np.linalg.inv(calcul_fbm_kernel(res.x[2], res.x[3], vec_t))
    cov3 = np.linalg.inv(calcul_fbm_kernel(res.x[4], res.x[5], vec_t))
    r1 = traj1*scale_factor-np.mean(traj1*scale_factor,axis=0)
    r2 =traj2*scale_factor-np.mean(traj2*scale_factor,axis=0)
    traj_back = np.matmul(np.linalg.inv(cov1+cov2+cov3),np.matmul(np.transpose(np.concatenate((cov1,cov2))),np.concatenate((r1,r2))))
    return res, traj_back

def VAF(epsilon = 1, time_round = 0.1, calib_px=1, calib_fr=1, calib_px_z = 1):
    # n_dim = np.shape(traj)[1]
    # V = np.diff(traj,axis=0)#/np.diff(vecT)  #velocity vector
    # ProdV = np.zeros((n_dim, np.shape(V)[0], np.shape(V)[0]))
    # matT, matTp = np.meshgrid(np.diff(vecT),np.diff(vecT))
    # for i in range(n_dim):
    #     matVx, matVy = np.meshgrid(V[:,i],V[:,i])
    #     ProdV[i,:,:] = (matVx/matT) * (matVy/matTp)
    # ProdV_fin = np.sum(ProdV,axis=0)
    # matT, matTp = np.meshgrid(vecT[1:],vecT[1:])
    # DeltaT = np.round((matT - matTp)/epsilon)*epsilon
    # getT = (DeltaT >= 0)*DeltaT + (DeltaT<0)*(-1)
    # df_vaf = pd.DataFrame(data={'ProdV': np.ndarray.flatten(ProdV_fin), 'Delta_t': np.ndarray.flatten(getT)})
    # df2 = df_vaf[df_vaf.Delta_t >= 0]
    # df_fin = df2.groupby('Delta_t').mean()
    # df_fin['ProdV'] = df_fin['ProdV']/df_fin['ProdV'][0]
    
    root = tkinter.Tk(className='Open trajectories', )
    file_tracer = askopenfilenames()
    if file_tracer[0][-3:] == 'xml':
        df_traj = {}
        for j in range(len(file_tracer)):
            df_traj[j] = load_xml_traj(file_tracer[j])
    elif file_tracer[0][-3:] == 'csv':
        df_traj = {}
        for j in range(len(file_tracer)):
            df_traj[j] = load_csv_traj(file_tracer[j])
    else:
        raise ValueError("Only xml or csv files!")
    file_save = asksaveasfile(title="Save results", initialfile=ntpath.basename(file_tracer[0][0:len(file_tracer[0])-4])+'_all_list_VAF', defaultextension=".json", filetypes=(("json file", "*.json"),))
    file_save.close()
    root.destroy()
    df_VAF_results = {}#pd.DataFrame(data={'traj_ID': [], 'VAF':[]})
    for i in range(len(df_traj)):
        print(str(i+1)+' / '+str(len(df_traj)))
        if df_traj[0].shape[1] == 3:
            n_dim = 2
            traj_exp = np.array([(df_traj[i][df_traj[i].keys()[1]]*calib_px).tolist(), (df_traj[i][df_traj[i].keys()[2]]*calib_px).tolist()]).T
        else:
            n_dim = 3
            traj_exp = np.array([(df_traj[i][df_traj[i].keys()[1]]*calib_px).tolist(), (df_traj[i][df_traj[i].keys()[2]]*calib_px).tolist(), (df_traj[i][df_traj[i].keys()[3]]*calib_px_z).tolist()]).T
        vec_t = np.array([(df_traj[i][df_traj[i].keys()[0]]*calib_fr).tolist()]).T
        vec_t = vec_t[:,0]
        #V = np.diff(traj_exp,axis=0)#velocity between adjacent frames  #velocity vector
        V = traj_exp[epsilon:]-traj_exp[0:-epsilon] #velocity between frames with epsilon frames interval
        ProdV = np.zeros((n_dim, np.shape(V)[0], np.shape(V)[0]))
        #matdiffT, matdiffTp = np.meshgrid(np.diff(vec_t),np.diff(vec_t)) #time intervals matrix
        matdiffT, matdiffTp = np.meshgrid(vec_t[epsilon:]-vec_t[0:-epsilon],vec_t[epsilon:]-vec_t[0:-epsilon])
        for j in range(n_dim):
            matVx, matVy = np.meshgrid(V[:,j],V[:,j])
            ProdV[j,:,:] = (matVx/matdiffT) * (matVy/matdiffTp)
        ProdV_fin = np.sum(ProdV,axis=0)
        #matT, matTp = np.meshgrid(vec_t[1:],vec_t[1:])
        matT, matTp = np.meshgrid(vec_t[epsilon:],vec_t[epsilon:])
        DeltaT = np.round((matT - matTp)/time_round)*time_round
        getT = (DeltaT >= 0)*DeltaT + (DeltaT<0)*(-1)
        df_vaf = pd.DataFrame(data={'ProdV': np.ndarray.flatten(ProdV_fin), 'Delta_t': np.ndarray.flatten(getT)})
        df2 = df_vaf[df_vaf.Delta_t >= 0]
        df_fin = df2.groupby('Delta_t').mean()
        df_fin['ProdV'] = df_fin['ProdV']/df_fin['ProdV'][0]
        df_VAF_results[ntpath.basename(file_tracer[i][0:len(file_tracer[i])-4])] = {'Lag':np.array(df_fin.index).tolist(), 'VAF':np.array(df_fin['ProdV']).tolist(), 'epsilon (frames)':epsilon}
        
        #res.x[0] = res.x[0]*calib_px**2*calib_fr**(-res.x[1])
        #df_gpytool = pd.concat([df_gpytool, pd.DataFrame(data={'traj_ID': [ntpath.basename(file_tracer[i][0:len(file_tracer[i])-4])], 'alpha':[res.x[1]], 'D (um2/s**a)':[res.x[0]], 'traj_length':[len(vec_t)]})])
    with open(file_save.name, "w") as outfile: 
        json.dump(df_VAF_results, outfile)

def plot_VAF():
    root = tkinter.Tk(className='Open trajectories', )
    file_tracer = askopenfilenames()
    root.destroy()
    try:
        with open(file_tracer[0]) as json_file:
            data = json.load(json_file)
    except:
        print("Only JSON file accepted!")
    df_vaf = pd.DataFrame(data={'Lag': [], 'VAF': []})
    for i in data.keys():
        df_vaf = pd.concat([df_vaf, pd.DataFrame(data={'Lag': data[i]['Lag'], 'VAF': data[i]['VAF']})])
    epsilon = data[i]['epsilon (frames)']
    df_mean = df_vaf.groupby('Lag').mean()
    ax = df_mean.plot(title=ntpath.basename(file_tracer[0])+', epsilon = '+str(epsilon), marker='x', markersize=4, linestyle='none')
    ax.set_ylabel('normalized VAF')
    ax.set_xlabel('Lag (s)')
    
def Gaussianity_plot(delta = [1, 5, 10, 15, 20], calib_fr = 1, bin_hist = 100):
    root = tkinter.Tk(className='Open trajectories', )
    file_tracer = askopenfilenames()
    root.destroy()
    df_csv = pd.read_csv(file_tracer[0])
    df_gauss = pd.DataFrame(data={'traj_ID': [], 'alpha':[], 'D (um2/s**a)':[], 'traj_length':[], 'x_dis':[], 'y_dis':[], 'delta':[]})
    for traj_name in df_csv['traj_ID']:
        if traj_name[-3:] == 'xml':
            df_traj = load_xml_traj(traj_name)
        elif traj_name[-3:] == 'csv':
            df_traj = load_csv_traj(traj_name)
        alpha = float(df_csv[df_csv['traj_ID']==traj_name]['alpha'])  #remove [0]
        D = float(df_csv[df_csv['traj_ID']==traj_name]['D (um2/s**a)'])
        df_traj.columns = range(df_traj.shape[1])
        for k in delta:
            diffT = np.round((np.array(df_traj[0][k:])-np.array(df_traj[0][0:-k]))/(k*calib_fr))==1 #detect missing frames: (example) frames 3 & 4 are missing, serie jumps from frame 2 to frame 5 and would consider this is one frame delta, which is wrong
            diffX = (np.array(df_traj[1][k:])-np.array(df_traj[1][0:-k]))/np.sqrt(2*D*(k*calib_fr)**alpha)
            diffY = (np.array(df_traj[2][k:])-np.array(df_traj[2][0:-k]))/np.sqrt(2*D*(k*calib_fr)**alpha)
            df_gauss = pd.concat([df_gauss, pd.DataFrame(data={'traj_ID': [traj_name]*len(diffX[diffT]), 'alpha':[alpha]*len(diffX[diffT]), 'D (um2/s**a)':[D]*len(diffX[diffT]), 'traj_length':[len(df_traj)]*len(diffX[diffT]), 'x_dis':diffX[diffT], 'y_dis':diffY[diffT], 'delta':[k]*len(diffX[diffT])})])
    #ax = df_gauss.hist(['x_dis','y_dis'], by='delta', bins=200,log=True,density=True,alpha=0.5)    
    df_gauss_histX = pd.DataFrame(data={'X':[], 'bins':[], 'delta':[]})
    df_gauss_histY = pd.DataFrame(data={'Y':[], 'bins':[], 'delta':[]})
    figTemp = plt.figure('temp')
    for k in delta:
        Xvalues, Xbins, _ = plt.hist(df_gauss[df_gauss['delta']==k]['x_dis'], bins=bin_hist, density = True)
        Xcent_bin = (Xbins[:-1]+Xbins[1:])/2
        df_gauss_histX = pd.concat([df_gauss_histX, pd.DataFrame(data={'X':Xvalues, 'bins':Xcent_bin, 'delta':[k]*len(Xcent_bin)})])
        Yvalues, Ybins, _ = plt.hist(df_gauss[df_gauss['delta']==k]['y_dis'], bins=bin_hist, density = True)
        Ycent_bin = (Ybins[:-1]+Ybins[1:])/2
        df_gauss_histY = pd.concat([df_gauss_histY, pd.DataFrame(data={'Y':Xvalues, 'bins':Ycent_bin, 'delta':[k]*len(Ycent_bin)})])
    #df_gauss_histX.plot(kind = 'scatter', x = 'bins', y = 'X')
    plt.close('temp')
    groups = df_gauss_histX.groupby('delta')

# Plot
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.bins, group.X, marker='x', linestyle='', ms=4, label=name)
    ax.set_yscale('log')
    ax.plot(np.linspace(-3,3,1000),np.exp(-0.5*np.linspace(-3,3,1000)**2)/np.sqrt(2*np.pi),'k-', label='standard Gaussian distrib.')
    plt.xlabel('FBM normalized X displacements')
    plt.ylabel('Log-density')
    plt.title(ntpath.basename(file_tracer[0]))
    ax.legend()
    plt.show()
    
    groupsY = df_gauss_histY.groupby('delta')

# Plot
    figY, axY = plt.subplots()
    axY.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groupsY:
        axY.plot(group.bins, group.Y, marker='x', linestyle='', ms=4, label=name)
    axY.set_yscale('log')
    axY.plot(np.linspace(-3,3,1000),np.exp(-0.5*np.linspace(-3,3,1000)**2)/np.sqrt(2*np.pi),'k-', label='standard Gaussian distrib.')
    plt.xlabel('FBM normalized X displacements')
    plt.ylabel('Log-density')
    plt.title(ntpath.basename(file_tracer[0]))
    axY.legend()
    plt.show()
        