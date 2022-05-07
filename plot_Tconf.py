"""
Script to plot results to various experiments. Reads in csv result file, typically
storing configurational / kinetic temperatures versus time as obtained by the sampling
schemes in this repository. Execute cells as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator
import csv
from cycler import cycler
from matplotlib import cm
import matplotlib as mp

plt.close('all')
plt.rcParams.update({'font.size': 35})
plt.rc('legend', fontsize=35)    # legend fontsize
plt.rcParams['axes.grid'] = True
plt.rc('lines', linewidth=3)
plt.rcParams['axes.titlepad'] = 0
mp.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

#%%

name = "/home/rene/PhD/Research/Code/Integrators/BNN/Tconf_kin_"    # head of file name to be read

## params that were sampled over
## for n sampled parameters, the files hold 2n+3 cols 
## (one for each param for both Tconf and Tkin, one for global values of Tconf and Tkin, and one for iterations)
params = ["W2", "b2"]
n = len(params)
n_cols = 2*(n+1)+1


#%% OBABO vs h, global Ts 

filename = name + "OBABO_h"
hs = ["0.01"]    # h in file names to open
col_list = [3,6]    # cols to plot
label_cores = ["Tconf", "Tkin"]    # content of cols, becomes parts of plot labels
title = r"Partial BNN, Last Layer Sampling, Spiral Data, 1000pts, OBABO, T=5"    # plot title

fig, ax = plt.subplots() 
for h in hs:
    X_meas = []
    with open(filename + h) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            n_axis = [int(row[0])]
            vals = [float(val) for val in row[1::]]
            vals = n_axis + vals
            X_meas += [vals]
    X_meas = np.array(X_meas)    
    
    labels = "h=" + str(h) + ", " 
    
    #Plot various cols
    for (idx,col) in enumerate(col_list):
        ax.plot(X_meas[:,0], X_meas[:,col], label = labels+label_cores[idx] )
    
ax.set_xlabel(r"$N_{{Samples}}$")
plt.tight_layout()
plt.title(title)
ax.legend()

#%% Plot OBABO with/ w/o grad noise

filename = name + "OBABO_h0.01"
batch_sfx = ["", "_gradB500", "_gradB50"]    # grad noise suffix in file names to open (empty string for full gradient)
batch_labels = ["100%", "50%", "5%"]
col_list = [3,6]    # cols to plot
label_cores = ["Tconf", "Tkin"]  # content of cols, becomes parts of plot labels
title = r"Partial BNN, Last Layer Sampling, Spiral Data, 1000pts, OBABO, T=1"    # plot title

fig, ax = plt.subplots() 
for (sfx, B_label) in zip(batch_sfx, batch_labels):
    X_meas = []
    with open(filename + sfx) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            n_axis = [int(row[0])]
            vals = [float(val) for val in row[1::]]
            vals = n_axis + vals
            X_meas += [vals]
    X_meas = np.array(X_meas)    
    
    labels = "B=" + B_label + ", " 
    
    #Plot various cols
    for (idx,col) in enumerate(col_list):
        ax.plot(X_meas[:,0], X_meas[:,col], label = labels+label_cores[idx] )
    
ax.set_xlabel(r"$N_{{Samples}}$")
plt.tight_layout()
plt.title(title)
ax.legend()

#%% Plot Metropolized schemes for gradient noise correction

h = "0.01"
name_sfx = ["OBABO_h"+h+"_gradB50", "OMBABO_h"+h+"_L1_SF0_gradB50", "OMBABO_h"+h+"_L1_SF1_gradB50"]    # suffix in file names to open
labels = ["OBABO, h="+h+", ", "OMBABO, L=1, SF0, ", "OMBABO, L=1, SFR, "]
col_list = [3,6]    # cols to plot
label_cores = ["Tconf", "Tkin"]    # content of cols, becomes parts of plot labels
title = r"Partial BNN, Last Layer Sampling, Spiral Data, 1000pts, T=1, B=5%"    # plot title

fig, ax = plt.subplots() 
for (sfx, label_meth) in zip(name_sfx, labels):
    X_meas = []
    with open(name + sfx) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            n_axis = [int(row[0])]
            vals = [float(val) for val in row[1::]]
            vals = n_axis + vals
            X_meas += [vals]
    X_meas = np.array(X_meas)    
    
    #Plot various cols
    for (idx,col) in enumerate(col_list):
        ax.plot(X_meas[:,0], X_meas[:,col], label = label_meth+label_cores[idx] )
    
ax.set_xlabel(r"$N_{{Samples}}$")
plt.tight_layout()
plt.title(title)
ax.legend()
