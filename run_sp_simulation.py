import numpy as np
import subprocess, os, glob, shutil
from shutil import copyfile
import pickle
import matplotlib.pyplot as plt
import math
import sys
import os.path
import time
import pandas as pd
import h5py
import Rate_Curve_Visualization

def calc_WI_formula_2d_isotropy_perm(k,size,skin_factor=0):
    # may need some additional unit conversion later
    cdarcy = 1#0.008527 # cdarcy should be done inside simulator well model not here
    pi = 3.1415926
    h = 32.8
    dx = (size/h)**0.5
    dy = dx
    r0 = 0.28*((k/k)**0.5*dx**2+(k/k)**0.5*dy**2)**0.5/((k/k)**0.25+(k/k)**0.25)
    rperf = 0.3048/2
    wi = 2*pi*cdarcy*k*h/(np.log(r0/rperf)+skin_factor)
    return wi

# additional, calculate the well index for each well based on the wellblock perm and the size of the wellblock
def calculate_wellindex(config_idx,num_cell,num_inj,num_prod,wellLoc,volume,perm):
    WI = [] # size of WI is equal to num_inj+num_prod
    # Loop through all well and extract corresponding perm and size info
    # Then, for each perm and size calc the WI correspondingly
    for inj in range(num_inj):
        curr_well = wellLoc[config_idx+1][0][inj] #0-based
        k = perm[curr_well]
        size = volume[curr_well]
        skin_factor = 0 # may change later
        WI.append(calc_WI_formula_2d_isotropy_perm(k,size,skin_factor))
    for prd in range(num_prod):
        curr_well = wellLoc[config_idx+1][1][prd] #0-based
        k = perm[curr_well]
        size = volume[curr_well]
        skin_factor = 0 # may change later
        WI.append(calc_WI_formula_2d_isotropy_perm(k,size,skin_factor))
    return WI

def Write_Well_Loc(wellLoc,curr_dir,num_cell,num_inj,num_prod,inj_bhps,prod_bhps):
    # first read the perm and block size for the specific case from "PEBI_PERM.dat" and "Volume.dat"
    fileName = curr_dir+'/Volume.DAT'
    volume = np.zeros(num_cell)
    with open(fileName) as f:
        lines = f.readlines()
    for i in range(len(lines)-2):
        volume[i] = float(lines[i+1])

    fileName = curr_dir+'/PEBI_PERM.dat'
    perm = np.zeros(num_cell)
    with open(fileName) as f:
        lines = f.readlines()
    for i in range(len(lines)-2):
        perm[i] = float(lines[i+1])
    
    config = 0
    WI = calculate_wellindex(config,num_cell,num_inj,num_prod,wellLoc,volume,perm)
    
    fileName = 'Well.DAT'
    subprocess.run(["rm",fileName],capture_output=True)
    f=open(fileName,"w")
    f.write("WELSPECS\n")
    for inj in range(num_inj):
        f.write("INJ"+str(inj+1)+"	GROUP1	"+str(wellLoc[config+1][0][inj]+1)+" 1 * /\n")
    for prd in range(num_prod):
        f.write("PRD"+str(prd+1)+"	GROUP1	"+str(wellLoc[config+1][1][prd]+1)+" 1 * /\n")
    f.write("/\n")
    f.write("\n")
    f.write("COMPDAT\n")
    f.write("-- name	cell	idk	idk	idk	open	idk	WI	rad\n")
    for inj in range(num_inj):
        f.write("INJ"+str(inj+1)+"	"+str(wellLoc[config+1][0][inj]+1)+"	1	1	1	OPEN	1*	"+str(WI[inj])+"	1*	/\n")
    for prd in range(num_prod):
        f.write("PRD"+str(prd+1)+"	"+str(wellLoc[config+1][1][prd]+1)+"	1	1	1	OPEN	1*	"+str(WI[prd+num_inj])+"	1*	/\n")
    f.write("/\n")
    f.write("\n")
    f.write("WELLSTRE\n")
    for inj in range(num_inj):
        f.write("INJ"+str(inj+1)+" 1 0 0 /\n")
    f.write("/\n")
    f.write("\n")
    f.write("WCONINJE\n")
    for inj in range(num_inj):
        f.write("INJ"+str(inj+1)+" WATER OPEN BHP 2* "+str(inj_bhps[inj])+"/\n")
    f.write("/\n")
    f.write("\n")
    f.write("WCONPROD\n")
    for prd in range(num_prod):
        f.write("PRD"+str(prd+1)+"  OPEN BHP 5* "+str(prod_bhps[prd])+" /\n")
    f.write("/\n")
    f.close()

def Run_ADGPRS(curr_dir):
    # apply for sdev cpu resources
    num_core = 1

    # remove previous output files
    fileName = "gprs.log.txt"
    filePath = curr_dir+"/"+fileName
    if os.path.exists(filePath):
        subprocess.run(["rm","-r",filePath])
        
    fileName = "Output.rates.txt"
    filePath = curr_dir+"/"+fileName
    if os.path.exists(filePath):
        subprocess.run(["rm","-r",filePath])

    fileName = "Output.vars.txt"
    filePath = curr_dir+"/"+fileName
    if os.path.exists(filePath):
        subprocess.run(["rm","-r",filePath])
   
    fileName = "Output.vars.h5"
    filePath = curr_dir+"/"+fileName
    if os.path.exists(filePath):
        subprocess.run(["rm","-r",filePath])
  
    fileName = "Output.wells"
    filePath = curr_dir+"/"+fileName
    if os.path.exists(filePath):
        subprocess.run(["rm","-r",filePath])
   
    # run batch.sh by using sbatch
    subprocess.run(["sbatch","batch.sh",str(num_core)],capture_output=True)
    return

def Read_ADGPRS_Var(curr_dir):
    adgprs_output = os.path.join(curr_dir, 'Output.vars.h5')

    fid = h5py.File(adgprs_output, 'r')
    Sw = np.array(fid.get('FLOW_CELL')['sat_np=1'])
    P = np.array(fid.get('FLOW_CELL')['pres'])
    fid.close()
    
    return P,Sw

def load_simulation_state_variables(curr_dir,num_inj,num_prod):
    # first, use prepared util functions to load rates and vars
    p,sw = Read_ADGPRS_Var(curr_dir) 
    # next, modify the data to be aligned with previous versions 
    P = np.transpose(p[:,1:]); Sw = np.transpose(sw[:,1:]) # specific change for adgprs as it outputs 1 more timestep than requested
    return P, Sw

def calc_WI_formula_2d_isotropy_perm_unit_correction(k,size,skin_factor=0):
    # may need some additional unit conversion later
    cdarcy = 0.008527
    pi = 3.1415926
    h = 32.8
    dx = (size/h)**0.5
    dy = dx
    r0 = 0.28*((k/k)**0.5*dx**2+(k/k)**0.5*dy**2)**0.5/((k/k)**0.25+(k/k)**0.25)
    rperf = 0.3048/2
    wi = 2*pi*cdarcy*k*h/(np.log(r0/rperf)+skin_factor)
    return wi

def extract_rates_at_certain_time_steps(time, rate, target_time, num_ts, keyword):
    count = 0
    rate_ts = []
    for j in range(time.shape[0]):
        if target_time[count] == time[j]:
            rate_ts.append(rate[keyword][0,j])
            count += 1
            if count == num_ts:
                rate_ts = np.array(rate_ts)
                return rate_ts
    rate_ts = np.array(rate_ts)
    return rate_ts

def Read_ADGPRS_Rate(curr_dir,well_ids):   
    rate = {}
    sub_file_path = os.path.join(curr_dir, 'Output.rates.txt')

    data = pd.read_csv(sub_file_path, delim_whitespace=True)
   #     continue
    for well_id in well_ids:
        if well_id + ':WAT' not in rate:
            rate[well_id + ':WAT'], rate[well_id + ':OIL'] = [], []
        rate[well_id + ':WAT'].append(data[well_id + ':WAT'].values)
        rate[well_id + ':OIL'].append(data[well_id + ':OIL'].values)
    for key in rate:
        rate[key] = np.stack(rate[key], axis = 0)
    
    return data['Day'], rate 

def read_simulation_rates(cwd,num_cell,num_inj,num_prod,num_ts,wellLoc,time_step):
    well_ids = []
    for i in range(num_inj):
        well_ids.append('INJ'+str(i+1))
    for i in range(num_prod):
        well_ids.append('PRD'+str(i+1))
        
    inj_rate = np.zeros((num_inj,num_ts))
    prd_wrate = np.zeros((num_prod,num_ts))
    prd_orate = np.zeros((num_prod,num_ts))
    
    # the time point we are interested in [time_step,2*time_step,...,n*time_step] size same as num_ts
    target_time = []
    for i in range(num_ts):
        target_time.append(time_step*(i+1))

    # extract all rates info from ADGPRS
    time, rate =  Read_ADGPRS_Rate(cwd,well_ids)
    for i in range(num_inj):
        inj_rate[i] = extract_rates_at_certain_time_steps(time, rate,target_time,num_ts,'INJ'+str(i+1)+':WAT')
    for i in range(num_prod):
        prd_orate[i] = extract_rates_at_certain_time_steps(time, rate,target_time,num_ts,'PRD'+str(i+1)+':OIL')
        prd_wrate[i] = extract_rates_at_certain_time_steps(time, rate,target_time,num_ts,'PRD'+str(i+1)+':WAT')
    
    return inj_rate,prd_wrate,prd_orate

def checkDist(index_new,index,XYZ,threshold):
    dist = ((XYZ['X'][index_new]-XYZ['X'][index])**2+(XYZ['Y'][index_new]-XYZ['Y'][index])**2++(XYZ['Z'][index_new]-XYZ['Z'][index])**2)**0.5
    return dist>threshold

def find_unstructured_cell_index_from_X_Y(x,y,XYZ):
    min_dist = 1e5
    cell_idx = -1
    for i in range(len(XYZ)):
        curr_dist = ((x-XYZ['X'][i])**2+(y-XYZ['Y'][i])**2)**0.5
        if curr_dist < min_dist:
            min_dist = curr_dist
            cell_idx = i
    return cell_idx # 0 -based cell_idx
################################################## main function #########################################################
def run_sp_simulation_main(infile):

    start_time = time.time()

    # note: the input well loc should be 0-based (0-8099) but the output well loc in Well.DAT should be 1-based (1-8100)
    num_cell = 6045
    num_inj=5
    inj_bhps=[]
    num_prod=5
    prod_bhps=[]
    num_ts = 50
    time_per_step = 30
    min_well_dist = 100
    num_well = num_inj+num_prod
    cwd = os.getcwd()
    XYZ = pd.read_csv(cwd+"/XYZ.in", delim_whitespace=True)

    # step 1. update the well location based on the input
    input_infos = np.loadtxt(infile) # first 2xn_w [X,Y] pairs for the n_w well locations, last n_w for the injectors and producers BHPs
    wellLoc = {}
    wellLoc_load = []

    for inj in range(num_inj):
        x = input_infos[inj*2]
        y = input_infos[inj*2+1]
        wellLoc_load.append(find_unstructured_cell_index_from_X_Y(x,y,XYZ))
        inj_bhps.append(input_infos[2*num_well+inj])
    for prd in range(num_prod):
        x = input_infos[2*num_inj+prd*2]
        y = input_infos[2*num_inj+prd*2+1]
        wellLoc_load.append(find_unstructured_cell_index_from_X_Y(x,y,XYZ))
        prod_bhps.append(input_infos[2*num_well+num_inj+prd])

    wellLoc[1] = [[wellLoc_load[0], wellLoc_load[1], wellLoc_load[2], wellLoc_load[3], wellLoc_load[4]], [wellLoc_load[5], wellLoc_load[6], wellLoc_load[7], wellLoc_load[8], wellLoc_load[9]]]

    Write_Well_Loc(wellLoc,cwd,num_cell,num_inj,num_prod,inj_bhps,prod_bhps)
    # step 101. before anything we need check if well distance is reasonable, if not just return NPV = 0
    is_well_dist = True
    wells = []
    for curr_idx in range(num_inj):
        wells.append(wellLoc[1][0][curr_idx])
    for curr_idx in range(num_prod):
        wells.append(wellLoc[1][1][curr_idx])
    for curr_idx in range(num_inj+num_prod):
        for new_idx in range(curr_idx):
            if not checkDist(wells[curr_idx],wells[new_idx],XYZ,min_well_dist):
                is_well_dist = False

    if not is_well_dist:
        output = [-1000.0]
        np.savetxt(outfile,output)
    else:
        # step 2. run the ADGPRS simulation 
        Run_ADGPRS(cwd)

        # step 3. hold and wait till ADGPRS finished then read in relevant variables
        fileName = "Output.vars.h5"
        filePath = cwd+"/"+fileName
            # first wait this to be created
        while not os.path.exists(filePath):
            time.sleep(1)
            # then wait until this file no more updating
        is_sim_running = True
        while(is_sim_running):
            curr_modification_time = os.path.getmtime(filePath)
            time.sleep(20) 
            next_modification_time = os.path.getmtime(filePath)
            if curr_modification_time == next_modification_time:
                is_sim_running = False # it is finished if this file not updating anymore 

    return
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
