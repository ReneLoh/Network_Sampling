
"""
This script tests Metropolized Langevin Samplers on Bayesian posteriors of small pretrained neural networks.
First a network is trained conventionally on 2-dimensional (for now) binary classification tasks.
Then a subset of the model parameters (currently the last layer) is sampled over in a Bayesian manner.
We track two temperatures as observables to examine sampling quality, which are printed to a file.

When sampling over K network parameters, the file contains 2K+3 columns: 
    1:          time
    next K:     configurational temperatures per param
    K+2:        global configurational temperature (i.e. averaged over all params)
    next K:     kinetic temperatures per param
    2K+3:       global kinetic temperature

The script uses mpi to draw N independent sampling trajectories to average over. At the moment, each worker
needs to perform the pretraining (not an issue as we are using small models). The only inter-process communication
happens at the very end for the averaging.

For now, we are interested in the behavior of the different samplers on the network posteriors in particular when
using stochastic gradients in the dynamical parts of the schemes. We hope to use these samplers in conjunction with stochastic
gradients in order to allow for more efficient Bayesian neural network training.
"""

import sys
import numpy as np
import csv as csv
import torch
sys.path.insert(0, "/home/rene/PhD/Research/Code/General_Neural_Network_Scripts")   # path to the following two modules
from torch_network_architectures import Net           # in repository "General_Neural_Network_Scripts"
from torch_train_routines import train_model_classi2  # in repository "General_Neural_Network_Scripts"
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from network_sampling import OBABO, MOBABO, OMBABO


#### For HPC Cluster Cirrus (uses custom mpi4py environment) ####

# import mpi4py.rc
# mpi4py.rc.initialize = False   # needed because Cirrus expects manual initialization
# from mpi4py import MPI
# MPI.Init()

####


from mpi4py import MPI       # comment out on Cirrus


comm = MPI.COMM_WORLD        # init mpi

rank = comm.Get_rank()       # rank is used as random seed
nr_proc = comm.Get_size()    # no. of processes



#%% SAMPLING PARAMS

sys.argv = [sys.argv[0]]
sys.argv += [3,1,0,0.01,200,1000]

method = int(sys.argv[1])    # method choice in {1,2,3}  (1=OBABO, 2=MOBABO, 3=OMBABO)
L = int(sys.argv[2])         # positive int, no. of integration steps within sampler
SF_mode = int(sys.argv[3])   # sign flip, 1=True, 0=False
h = float(sys.argv[4])       # step size
Ns = int(sys.argv[5])        # no. of samples taken
B = int(sys.argv[6])         # gradient batch size for sampling
T = 1                        # sampling temperature
gam = 1                      # Langevin friction
tavg = True					 # time average after ensemble average?
n = int(1e4) 				 # if tavg=True, time average over last n values
ndist = Ns//200				 # write out any ndist-th result entry to file 
n_meas = 1                   # collect sample any n_meas iterations 

#%% ARCHITECTURE + TRAINING PARAMS

nodenr = [2, 500,  1]   # nodes per layer
net = Net               # network model (see imports)
std = 0.5               # standard deviation for network param init
bs =  50                # batch size used for training
epochs = 1000           # no. training epochs
meas_freq = 50          # take loss/accuracy measurement any meas_freq steps
reps = 1                # no. of independent training runs
seed = 1                # random seed for training

#SGD / Adam parameters
eta = 0.05              # learning rate
mom = 0.99              # momentum (for SGDm)

#%% LOAD DATA (trigonometric or spiral, 2dim binary classification)

folder = "spiral_data/"
data_name = "spiral"    # "spiral" or "trigo"

Xtrain = torch.Tensor( np.load( folder + data_name + "_train_features.npy" ) )
ytrain = torch.Tensor( np.load( folder + data_name + "_train_labels.npy" ) )
Xtest = torch.Tensor( np.load( folder + data_name + "_test_features.npy" ) )
ytest = torch.Tensor( np.load( folder + data_name + "_test_labels.npy" ) )

# view the data set
# print("Loaded Data has shape {} (features) and {} (labels).".format(Xtrain.size(), ytrain.size()))
# N=500
# plt.scatter(Xtrain.numpy()[0:N-1,0], Xtrain.numpy()[0:N-1,1], s=3, label="Class 1", color="g")
# plt.scatter(Xtrain.numpy()[N:2*N-1,0], Xtrain.numpy()[N:2*N-1,1], s=3, label="Class 2", color="b")

train_set = torch.utils.data.TensorDataset(Xtrain, ytrain)   # turns it to TensorDataset
test_set = torch.utils.data.TensorDataset(Xtest, ytest)

# %% TRAINING / OPTIMIZATION

epochs_train = np.arange(0, epochs+1)
epochs_test = np.arange(0, (epochs//meas_freq + 1)*meas_freq, meas_freq)  # epochs in which model is tested on test set

print("starting training...")
start_time = time.time()

batches_train = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True) # allows iterating
batches_test = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True)   # over batches

loss_train = np.zeros(epochs+1)             # store loss
accu_train = np.zeros(epochs+1)             # store accuracies
loss_test = np.zeros(epochs//meas_freq+1)
accu_test = np.zeros(epochs//meas_freq+1)

for rep in range(0,reps):   # average over independent runs

    serial_net = net(nodenr,std)
    # optimizer = optim.SGD(serial_net.parameters(), lr=eta, weight_decay=0, momentum=mom)
    # optimizer = SGDm_original(serial_net.parameters(), lr=eta, weight_decay=0, momentum=mom, p_scaling=0.1)
    optimizer = optim.Adam(serial_net.parameters(), lr=eta, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    (loss_train_current, accu_train_current, loss_test_current, accu_test_current) = train_model_classi2(serial_net, optimizer, 
                                                                                                         F.binary_cross_entropy, 
                                                                                                         epochs=epochs, 
                                                                                                         trainloader=batches_train, 
                                                                                                         scheduler=None, 
                                                                                                         testloader=batches_test, eval_freq=meas_freq)
    
    loss_train += loss_train_current
    accu_train += accu_train_current
    loss_test += loss_test_current
    accu_test += accu_test_current


train_loss_total = loss_train/reps    # compute averages
train_accu_total = accu_train/reps 
test_loss_total = loss_test/reps
test_accu_total = accu_test/reps      


end_time = time.time()
print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
      .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/epochs))



#%% SAMPLING

# now sample over posterior of parts of the pretrained model and track observables (to examine sampling quality)

batches_train = torch.utils.data.DataLoader(train_set, batch_size=B, shuffle=True)    # data loader of batch size used in sampling

torch.manual_seed(rank)
model_copy = copy.deepcopy(serial_net)    # sample over copy of pretrained network

# choose sampling method and specify name for file to be printed 
if method == 1:
    (Tconfs, Tkins) = OBABO(model_copy, [model_copy.lin2.bias, model_copy.lin2.weight], F.binary_cross_entropy, batches_train, 
                            Ns, h, T, gam, n_meas)
    file_name = "_OBABO_h"+str(h) 

elif method == 2:
    (Tconfs, Tkins) = MOBABO(model_copy, [model_copy.lin2.bias, model_copy.lin2.weight], F.binary_cross_entropy, batches_train, 
                             Ns, h, T, gam, L, SF_mode, n_meas)
    file_name = "_MOBABO_h"+str(h) + "_L"+str(L)+"_SF"+str(int(SF_mode))

elif method == 3:
    (Tconfs, Tkins) = OMBABO(model_copy, [model_copy.lin2.bias, model_copy.lin2.weight], F.binary_cross_entropy, batches_train, 
                             Ns, h, T, gam, L, SF_mode, n_meas)
    file_name = "_OMBABO_h"+str(h) + "_L"+str(L)+"_SF"+str(int(SF_mode))

else: 
    print("No proper method index!")

print("Rank ", rank, "finnished sampling!")
 
#%% POSTPROCESSING (average and print)

if rank==0: print("Trajectory average...")
Tconf_avg = comm.reduce(Tconfs, op=MPI.SUM, root=0)    # sum over trajectories
Tkin_avg = comm.reduce(Tkins, op=MPI.SUM, root=0)      # obtained by the workers


if rank == 0:
    Tconf_avg /= nr_proc    # completes trajectory average
    Tkin_avg /= nr_proc

    last_idx = 1     # first index >0 at which we print results to files.
                     # after this, steps increase by ndist (see below).
    
    # now time average
    if tavg == True:
        print("\nTime average...")
        
        for i in range(len(Tconf_avg)-1, -1, -1*ndist):    # iterate backwards (since at each considered index, 
                                                           # we use information from smaller indices)
            
            if i<=n-1:    # we average over previous n indices, 
                n=i       # but need to adjust n if current i becomes too small

            Tconf_avg[i] = torch.sum(Tconf_avg[i-n:i], axis=0)    # sum over last n values...
            Tkin_avg[i] = torch.sum(Tkin_avg[i-n:i], axis=0)
            Tconf_avg[i] /= n+1                                   # ... and average
            Tkin_avg[i] /= n+1
        
        last_idx = i 
    

    # WRITE FILE
    if B < len(batches_train.dataset): output_file = "Tconf_kin" + file_name + "_gradB"+str(B)    # add batch size to file name if it is smaller 
    else: output_file = "Tconf_kin" + file_name                                                   # than than whole data set
    
    with open(output_file, mode='w') as res_file:
        
        file_writer = csv.writer(res_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        file_writer.writerow([0] + [ format(tconf, '.8g') for tconf in Tconf_avg[0] ] +   # idx 0 might be ommitted otherwise
                                    [format(tkin, '.8g') for tkin in Tkin_avg[0]] )     
        
        for i in range(last_idx, len(Tconf_avg), ndist):
            file_writer.writerow([i*n_meas] + [ format(tconf, '.8g') for tconf in Tconf_avg[i] ] + 
                                              [ format(tkin, '.8g') for tkin in Tkin_avg[i] ] )  


MPI.Finalize    # terminate mpi
