import re
import pandas as pd
import numpy as np
def quoteparse(text):
    pat='("[^"]+"|\s+)'
    vals=re.split(pat, text)
    return [ val.strip('"')  for val in vals if len(val.strip())>0]

def read_colnames(f,line):
    cols=[]    
    while 'END HEADER' not in line.strip():        
        cols.append(line.strip())            
        line=f.readline()
#         print(line.strip())
    return cols
# def readrows(f,line):
#     allvals=[]
#     while line:
#         vals_str=line.strip().split()
#         vals=[float(val) for val in vals_str]
#         allvals.append(vals)
#         line=f.readline()
#     return allvals
def readrows(f,line):
    pat='[+-]?([0-9]*[.])?[0-9]+'
    allvals=[]
    while line:
        vals_str= quoteparse(line)
        vals=[]
        for val in vals_str:
            if re.match(pat,val):
                vals.append(float(val))
            else:
                vals.append(val)
#         print(len(vals),len(allvals[0]))
        if len(allvals)==0:
            allvals.append(vals)
        else:
#             print(len(vals),len(allvals[0]))
            if len(vals)==len(allvals[0]):
                allvals.append(vals)
            else:
                print('There is problem in file format please cross check wiith result')
        
        line=f.readline()
    return np.array(allvals)
def read_twocolfile(file,nheadrows=2,colnames=[]):
    cols=colnames  
    with open(file) as f:
        line=f.readline()
        if len(colnames)<2:
            cols=line.strip().split()
        for i in range(1,nheadrows):
            f.readline()        
        while line:
            
            line=f.readline()
#             print(line,readrows(f,line))
            
            values=readrows(f,line)
            break
#         print ( values[:,1])
        return {cols[i]:values[:,i] for i in range(len(cols))}

def read_f_w_beginheader(file):
    with open(file) as f:
        line=f.readline()
        while line:
    #         print(line)
            line=f.readline()
            if line.strip()=='BEGIN HEADER':
                line=f.readline()
                props=read_colnames(f,line)
                line=f.readline()
                values=readrows(f,line)
                break
    welltops={}
    for p in props:
        welltops[p]=[]
    for val in values:
        for v,p in zip(val,props):
            welltops[p].append(v)
    return pd.DataFrame(welltops)
def read_welltops(file):
    return read_f_w_beginheader(file)
def read_tworowfile(file,nheadrows=2,colnames=[]):
    print("in future this function will be depricated... \n use read_twocolfile instead")
    return read_twocolfile(file,nheadrows=2,colnames=[])

def read_chkt(file):
    return read_f_w_beginheader(file)
def read_dev(file):
    with open(file) as f:
        line=f.readline()
        while line:
    #         print(line)
            line=f.readline()
            if line[:2]=='#=':
                line=f.readline()
                props=line.strip().split()
                f.readline()
                values=readrows(f,f.readline())
                break
    return pd.DataFrame(values,columns=props)

# folder=r"D:\Ameyem\D11_inversion\well_logs\\"
# file=folder+'d11_welltops_payzones_220519_sai.dat'
# sand_topbots=['Sand-1_Top', 'Sand-1_Bottom']
# wt[wt.Surface.isin(sand_topbots)].Z

