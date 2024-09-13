import os
import os.path as osp
import pickle
from pickle import dump,load
import numpy as np
import math, time
import pandas as pd
import copy
import matplotlib.pyplot as plt 


def load_pvdo(root):
    data_file =root
    pvdo_table = np.loadtxt(data_file + '/PVDO.DAT', skiprows=1, comments='/')
    return pvdo_table


def load_swof(root):
    data_file =root
    swof_table = np.loadtxt(data_file + '/SWOF.DAT', skiprows=1, comments='/')
    return swof_table


def cal_kr(sw,swof_table):
    sw_table = swof_table[:, 0]
    krw_table = swof_table[:, 1]
    kro_table = swof_table[:, 2]
    krw = np.interp(sw, sw_table, krw_table)
    kro = np.interp(sw, sw_table, kro_table)
    
    swi = sw_table[0]
    sor = 1-sw_table[-1]
    if sw < swi:
        krw = 0
        kro = 1
    elif sw > 1-sor:
        krw = 1
        kro = 0
    return krw, kro


def cal_pvto(p, pvdo_table):
    p_table = pvdo_table[:, 0]
    bo_table = pvdo_table[:, 1]
    vo_table = pvdo_table[:, 2]
    bo = np.interp(p, p_table, bo_table)
    vo = np.interp(p, p_table, vo_table)
    return bo, vo


def cal_pvtw(p):
    pw_ref, bw_ref, cw, vw_ref, cv = 273.1832, 1.029, 4.599848e-5, 0.5, 0.0
    y = -cv * (p - pw_ref)
    vw = vw_ref / (1. + y + y*y/2.0)
    x = cw * (p - pw_ref)
    bw = bw_ref / (1. + x + x*x/2.0)
    return bw, vw


def cal_mobi(p, sw,root):
    swof_table = load_swof(root)
    krw, kro = cal_kr(sw,swof_table)
    pvdo_table = load_pvdo(root)
    bo, vo = cal_pvto(p, pvdo_table)
    bw, vw = cal_pvtw(p)
    mo = kro/(vo*bo)
    mw = krw/(vw*bw)
    return mw, mo


def cal_prod_rate(p, sw, bhp, wi, root):
    c =  1#0.008527 # since we use wi from processed_data and it is multiplied with the darcy factor in Build_Graph_Dataset_Validation no need to doulbe-done this!
    nt = len(p)
    orat = np.zeros((nt, ))
    wrat = np.zeros((nt, ))

    for i in range(nt):
        pt = p[i]
        swt = sw[i]
        bhpt = bhp[i]

        mw, mo = cal_mobi(pt, swt,root)
        orat[i] = c*wi * mo * (pt - bhpt)
        wrat[i] = c*wi * mw * (pt - bhpt)
    return wrat, orat


def cal_inj_rate(p, sw, bhp, wi, root):
    c =  1#0.008527
    nt = len(p)
    irate = np.zeros((nt, ))
    for i in range(nt):
        pt = p[i]
        swt = sw[i]
        bhpt = bhp[i]

        mw, mo = cal_mobi(pt, swt, root)
        irate[i] =  c*wi*(mo + mw)*(bhpt-pt)
    
    return irate