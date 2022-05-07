"""
Sampling methods used in this repository.

OBABO: Splitting scheme for Langevin Dynamics.

MOBABO: Metropolized version of OBABO, where Metropolis step happens
        after L OBABO steps.

OMBABO: Metropolized version of OBABO, where Metropolis step happens
        after L BAB steps, and the O steps are executed before the BAB steps
        and after the MH step.
        
All methods can use stochastic gradients and return kinetic and configurational
temperatures.
"""


import torch
import numpy as np
import sys
import time
import copy




def get_single_batch(dataloader, dataloader_iterator):
    """
    Returns a single batch from a given data loader and corresponding iterator.
    """
    try:
        features, targets = next(dataloader_iterator)
    
    except StopIteration:                              # in case iterator is at its end
        dataloader_iterator = iter(dataloader)         # it needs to be recreated
        features, targets = next(dataloader_iterator)
    
    return (features, targets)




def OBABO(model, param_list, criterion, dataloader, N, h, T, gamma, n_meas):
    """
    OBABO
    
    model:          (pretrained) neural network
    param_list:     list of references to model parameters to be sampled over
    criterion:      loss function 
    dataloader:     data loader object holding the data with specified batch size
    N:              number of iterations / samples to be taken
    h:              step size
    T:              temperature
    gamma:          Langevin friction parameter
    n_meas:         measure observables any n_meas iterations
    
    """

    lam = 0  # weight decay (i.e. prior variance, prior is Gaussian)
    
    grad_scale = len(dataloader.dataset) / dataloader.batch_size  # prefactor for sum of stoch. gradients

    # turn off gradients for model parameters that are not being sampled. 
    # IMPORTANT: all parameters not being sampled must be in earlier layers than sampled params.
    # otherwise, turning off gradients impedes proper gradient flow...
    for theta in list(model.parameters()):
        theta.requires_grad = False
    for theta in param_list:
        theta.requires_grad = True
        if theta.grad is not None:
            theta.grad.zero_()
    
    # data iterators to get small and full batches
    full_batch_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size = len(dataloader.dataset), shuffle=False)
    full_batch_iter = iter(full_batch_loader) 
    small_batch_iter = iter(dataloader)

    # calculate initial force
    (features, targets) = get_single_batch(dataloader, small_batch_iter)
    loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")   # unsqueeze for BCE
    loss.backward()

    # build momentum and force buffers
    momentum_list = []
    forces = []
    for theta in param_list:
        dummy = theta.detach()
        momentum_list.append(torch.zeros_like(dummy))
        forces.append(-1*grad_scale*theta.grad - lam*dummy)
        theta.grad.zero_() 

    # get initial Tconfs/Tkins
    Tconfs = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)   # stores Tconf for each sampled parameter, as well as global one
    Tkins = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)    # stores Tkins in same manner
    Nds = torch.zeros(len(param_list)+1)                                    # degrees of freedom, normalizes Tconfs/Tkins at the end
    
    (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
    loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")   # unsqueeze for BCE
    loss.backward()
    
    for (idx, theta) in enumerate(param_list):
        Nds[idx] = theta.numel()     # count degs. of freedom
        Tconfs[0,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
        Tkins[0,idx] = torch.sum(momentum_list[idx]**2)                               # and Tkin
        theta.grad.zero_()
    
    k = 1                   # row index of Tconfs array to be filled     
    

    # START SAMPLING
    
    a = np.exp(-gamma*h)            # OBABO constant
    
    print("starting sampling...")
    start_time = time.time()
    
    for i in range(0,N):    # N iterations
        
        # ABO steps
        with torch.no_grad():
            for (theta, p, force) in zip(param_list, momentum_list, forces):
                
                rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
                p.mul_(np.sqrt(a))
                p.add_(np.sqrt((1-a)*T) * rand + 0.5*h*force)     # O + B step
                
                theta.add_(h*p)    # A step

        # calculate forces
        (features, targets) = get_single_batch(dataloader, small_batch_iter)
        loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")   # unsqueeze for BCE
        loss.backward()

        for j in range(0, len(param_list)):
            forces[j] = -1 * grad_scale * param_list[j].grad - lam*param_list[j].detach()
            param_list[j].grad.zero_()  # prepare next backward() 
        
        # OB steps
        with torch.no_grad():
            for (theta, p, force) in zip(param_list, momentum_list, forces):      
                
                p.add_(0.5*h*force)    # B step
                rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
                p.mul_(np.sqrt(a))
                p.add_(np.sqrt((1-a)*T) * rand)    # O step

        # take measurement
        if i % n_meas == 0:         
            
            (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)  # get full gradients for observable calc.
            loss = criterion(model(features), targets.unsqueeze(1), reduction="sum") 
            loss.backward()
            
            for (idx,theta) in enumerate(param_list):
                Tconfs[k,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
                Tkins[k,idx] = torch.sum(momentum_list[idx]**2)      # Tkin
                theta.grad.zero_()
            k += 1
        
        
        if i % 100000 == 0:
            print("Sampling iteration ",i," done! \n")
            
       
    end_time = time.time()
    print("Sampling took {} seconds, i.e {} minutes."
          .format(end_time-start_time, (end_time-start_time)/60))    
   
    
    # compute global Tconf and normalize 
    Tconfs[:,-1] = torch.sum(Tconfs, dim=1)
    Tkins[:,-1] = torch.sum(Tkins, dim=1)
    Nds[-1] = torch.sum(Nds)
    Tconfs /= Nds
    Tkins /= Nds
    
    return (Tconfs, Tkins)




def MOBABO(model, param_list, criterion, dataloader, N, h, T, gamma, L, SF_mode, n_meas):
    """
    MOBABO
    
    model:          (pretrained) neural network
    param_list:     list of references to model parameters to be sampled over
    criterion:      loss function 
    dataloader:     data loader object holding the data with specified batch size
    N:              number of iterations / samples to be taken
    h:              step size
    T:              temperature
    gamma:          Langevin friction parameter
    L:              number of steps in the dynamical part
    SF_mode:        either 1 or 0; sign flip of momentum after Metropolis rejection
    n_meas:         measure observables any n_meas iterations
    
    """

    lam = 0    # weight decay (i.e. prior variance, prior is Gaussian)
    
    grad_scale = len(dataloader.dataset) / dataloader.batch_size    # prefactor for sum of stoch. gradients

    # turn off gradients for model parameters that are not being sampled. 
    # IMPORTANT: all parameters not being sampled must lie in earlier layers than sampled params.
    # otherwise, turning off gradients impedes proper gradient flow...
    for theta in list(model.parameters()):
        theta.requires_grad = False
    for theta in param_list:
        theta.requires_grad = True
        if theta.grad is not None:
            theta.grad.zero_()
    
    
    # data iterators to get small and full batches
    full_batch_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size = len(dataloader.dataset), shuffle=False)
    full_batch_iter = iter(full_batch_loader) 
    small_batch_iter = iter(dataloader)


    # calculate initial force
    (features, targets) = get_single_batch(dataloader, small_batch_iter)
    loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
    loss.backward()

    # build momentum and force buffers, as well as list holding "current" params
    param_list_curr = []
    momentum_list_curr = []
    momentum_list = []
    forces_curr = []
    forces = []
    for theta in param_list:
        dummy = theta.detach()
        param_list_curr.append(dummy.clone())                       # the current params don't need gradients. 
                                                                    # when they are copied into the network in case of rejection 
                                                                    # of the last sample, gradient tracking is activated by default.             
        momentum_list_curr.append(torch.zeros_like(dummy))
        forces_curr.append(-1*grad_scale*theta.grad - lam*dummy)
        theta.grad.zero_() 
    
    forces = copy.deepcopy(forces_curr)
    momentum_list = copy.deepcopy(momentum_list_curr)

    # get initial Tconfs/Tkins
    Tconfs = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)    # stores Tconf for each sampled parameter, as well as global one
    Tkins = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)     # stores Tkins in same manner
    Nds = torch.zeros(len(param_list)+1)                                     # degrees of freedom, normalizes Tconfs/Tkins at the end
    
    (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
    loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
    loss.backward()
    
    for (idx, theta) in enumerate(param_list):
        Nds[idx] = theta.numel()     # count degs. of freedom
        Tconfs[0,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )    # compute Tconf
        Tkins[0,idx] = torch.sum(momentum_list[idx]**2)                                # and Tkin
        theta.grad.zero_()
    
    k = 1          # row index of Tconfs/Tkins arrays to be filled     
    
    
    # get initial potential energy
    U0 = 0
    with torch.no_grad():
        for param0 in (param_list):
            U0 += torch.sum(param0**2)
    U0 = 1/2 * lam * U0 + loss.detach()
    
    
    # START SAMPLING
    
    a = np.exp(-gamma*h)     # OBABO constant
    acc = 0                  # acceptance rate
    rejected = False         # tracks whether Metropolis step rejected sample
    
    print("starting sampling...")
    start_time = time.time()
    
    #### outer loop ####
    for i in range(0,N):    # generate N samples
        
    
        # after rejection, reset proposed params to current params
        if rejected == True:
            with torch.no_grad():
                for idx in range(0, len(param_list_curr)):
                    param_list[idx].copy_(param_list_curr[idx])

            forces = copy.deepcopy(forces_curr)
            
            if SF_mode:    # sign flip
                momentum_list_curr = [-1*p for p in momentum_list_curr]
            momentum_list = copy.deepcopy(momentum_list_curr)
                
        
        kin_energy = 0
        U1 = 0
        
        #### inner loop ####
        for j in range(0,L):    # L integrator steps to get proposal
        
            # ABO steps
            with torch.no_grad():
                for (theta, p, force) in zip(param_list, momentum_list, forces):
                    
                    rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
                    p.mul_(np.sqrt(a))
                    p.add_(np.sqrt((1-a)*T) * rand)     # O step
                    
                    kin_energy -= torch.sum(p**2)
                    
                    p.add_(0.5*h*force)     # B step
                      
                    theta.add_(h*p)    # A step
          
            # calculate forces
            (features, targets) = get_single_batch(dataloader, small_batch_iter)
            loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
            loss.backward()

            for m in range(0, len(param_list)):
                forces[m] = -1 * grad_scale * param_list[m].grad - lam*param_list[m].detach()
                param_list[m].grad.zero_()    # prepare next backward() 
     
            # OB steps
            with torch.no_grad():
                for (theta, p, force) in zip(param_list, momentum_list, forces):      
                    
                    p.add_(0.5*h*force)    # B step
                    
                    kin_energy += torch.sum(p**2)
                    
                    rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
                    p.mul_(np.sqrt(a))
                    p.add_(np.sqrt((1-a)*T) * rand)     # O step


        #### end inner loop ####

        kin_energy *= 0.5

        # compute U1
        (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
        with torch.no_grad():
            for param in param_list:
                U1 += torch.sum(param**2)
            loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
        U1 = 1/2 * lam * U1 + loss
        
        
        #### MH Criterion ####
        
        MH = torch.exp( (-1/T) * (U1 - U0 + kin_energy) )
        
        if( torch.rand(1) < min(1., MH) ):    # accept sample

            if i % n_meas == 0:    # take measurement  
                (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)    # get full gradients
                loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")      # unsqueeze for BCE
                loss.backward()
                for (idx,theta) in enumerate(param_list):
                    Tconfs[k,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )    # compute Tconf
                    Tkins[k,idx] = torch.sum(momentum_list[idx]**2)                                # and Tkin
                    theta.grad.zero_()
                k += 1

            # update current params, forces, and momenta
            for j in range(0, len(param_list)):  
                param_list_curr[j] = param_list[j].detach().clone()
            forces_curr = copy.deepcopy(forces)
            momentum_list_curr = copy.deepcopy(momentum_list)
            U0 = U1.clone()

            rejected = False
            acc += 1

        else:    # reject sample

            if i % n_meas == 0:    # take measurement  
                Tconfs[k] = Tconfs[k-1]
                Tkins[k] = Tkins[k-1]
                k += 1

            rejected = True
        
        #### end MH Criterion ####
        

        if i % 100000 == 0:
            print("Sampling iteration ",i," done! \n")
            
    #### end outer loop ####


    end_time = time.time()
    print("Sampling took {} seconds, i.e {} minutes."
          .format(end_time-start_time, (end_time-start_time)/60))
    print("Acceptance rate: " + str(acc/N))
  

    # compute global Tconf and normalize 
    Tconfs[:,-1] = torch.sum(Tconfs, dim=1)
    Tkins[:,-1] = torch.sum(Tkins, dim=1)
    Nds[-1] = torch.sum(Nds)
    Tconfs /= Nds
    Tkins /= Nds

    return (Tconfs, Tkins)



def OMBABO(model, param_list, criterion, dataloader, N, h, T, gamma, L, SF_mode, n_meas):
    """
    OMBABO
    
    model:          (pretrained) neural network
    param_list:     list of references to model parameters to be sampled over
    criterion:      loss function 
    dataloader:     data loader object holding the data with specified batch size
    N:              number of iterations / samples to be taken
    h:              step size
    T:              temperature
    gamma:          Langevin friction parameter
    L:              number of steps in the dynamical part
    SF_mode:        either 1 or 0; sign flip of momentum after Metropolis rejection
    n_meas:         measure observables any n_meas iterations
    
    """

    lam = 0    # weight decay (i.e. prior variance, prior is Gaussian)
    
    grad_scale = len(dataloader.dataset) / dataloader.batch_size    # prefactor for sum of stoch. gradients


    # turn off gradients for model parameters that are not being sampled. 
    # IMPORTANT: all parameters not being sampled must lie in earlier layers than sampled params.
    # otherwise, turning off gradients impedes proper gradient flow...
    for theta in list(model.parameters()):
        theta.requires_grad = False
    for theta in param_list:
        theta.requires_grad = True
        if theta.grad is not None:
            theta.grad.zero_()
    
    
    # data iterators to get small and full batches
    full_batch_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size = len(dataloader.dataset), shuffle=False)
    full_batch_iter = iter(full_batch_loader) 
    small_batch_iter = iter(dataloader)


    # calculate initial force
    (features, targets) = get_single_batch(dataloader, small_batch_iter)
    loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")   # unsqueeze for BCE
    loss.backward()


    # build momentum and force buffers, as well as list holding "current" params
    param_list_curr = []
    momentum_list_curr = []
    momentum_list = []
    forces_curr = []
    forces = []
    for theta in param_list:
        dummy = theta.detach()
        param_list_curr.append(dummy.clone())                       # the current params don't need gradients. 
                                                                    # when they are copied into the network in case of rejection 
                                                                    # of the last sample, gradient tracking is activated by default.             
        momentum_list_curr.append(torch.zeros_like(dummy))
        forces_curr.append(-1*grad_scale*theta.grad - lam*dummy)
        theta.grad.zero_() 
        
    forces = copy.deepcopy(forces_curr)
    momentum_list = copy.deepcopy(momentum_list_curr)

    # get initial Tconfs/Tkins
    Tconfs = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)    # stores Tconf for each sampled parameter, as well as global one
    Tkins = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)     # stores Tkins in same manner
    Nds = torch.zeros(len(param_list)+1)                                     # degrees of freedom, normalizes Tconfs/Tkins at the end
    
    (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
    loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
    loss.backward()
    
    for (idx, theta) in enumerate(param_list):
        Nds[idx] = theta.numel()    # count degs. of freedom
        Tconfs[0,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )    # compute Tconf
        Tkins[0,idx] = torch.sum(momentum_list[idx]**2)                                # and Tkin
        theta.grad.zero_()
    
    k = 1    # row index of Tconfs array to be filled     
    
    
    # get initial potential energy
    U0 = 0
    with torch.no_grad():
        for (param0, p0) in zip(param_list, momentum_list):
            U0 += torch.sum(param0**2)
    U0 = 1/2 * lam * U0 + loss.detach()
    
    
    # START SAMPLING
    
    a = np.exp(-gamma*h)     # OBABO constant
    acc = 0                  # acceptance rate
    rejected = False         # tracks whether Metropolis step rejected sample
    
    print("starting sampling...")
    start_time = time.time()
    
    #### outer loop ####
    for i in range(0,N):    # generate N samples
        
        K0 = 0    # kinetic energy (to be computed after O step, before BAB stps)
        
        # O step
        with torch.no_grad():
            for p in momentum_list_curr:
                
                rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
                p.mul_(np.sqrt(a))
                p.add_(np.sqrt((1-a)*T) * rand)     # O step
                K0 += torch.sum(p**2)
        
        K0 *= 1/2
        
        
        # reset proposed params to current params
        if rejected == True:
            with torch.no_grad():
                for idx in range(0, len(param_list_curr)):
                    param_list[idx].copy_(param_list_curr[idx])

            forces = copy.deepcopy(forces_curr)
            if SF_mode:    # sign flip
                momentum_list_curr = [-1*p for p in momentum_list_curr]    # typically done within MH step,
                                                                           # but should be fine here as neither SF nor
                                                                           # O steps change invariant p-density.
        
        momentum_list = copy.deepcopy(momentum_list_curr)
              
       
        #### inner loop ####
        for j in range(0,L):    # L integrator steps to get proposal
        
            # AB steps
            with torch.no_grad():
                for (theta, p, force) in zip(param_list, momentum_list, forces):                    
                    
                    p.add_(0.5*h*force)    # B step
                    theta.add_(h*p)    # A step
                
                    
            # calculate forces
            (features, targets) = get_single_batch(dataloader, small_batch_iter)
            loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
            loss.backward()

            for m in range(0, len(param_list)):
                forces[m] = -1 * grad_scale * param_list[m].grad - lam*param_list[m].detach()
                param_list[m].grad.zero_()    # prepare next backward() 


            # B step
            with torch.no_grad():
                for (theta, p, force) in zip(param_list, momentum_list, forces):      
                    
                    p.add_(0.5*h*force)    # B step
             
                    
        #### end inner loop ####

        # compute U1, K1       
        U1 = 0
        K1 = 0
        (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
        with torch.no_grad():
            for (param, p) in zip(param_list, momentum_list):
                U1 += torch.sum(param**2)
                K1 += torch.sum(p**2)
            loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")   # unsqueeze for BCE
        U1 = 1/2 * lam * U1 + loss
        K1 *= 1/2
        
        
        #### MH Criterion ####
        
        MH = torch.exp( (-1/T) * (U1 - U0 + K1 - K0) )

        if( torch.rand(1) < min(1., MH) ):    # accept sample

            if i % n_meas == 0:    # take measurement  
                (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)    # get full gradients
                loss = criterion(model(features), targets.unsqueeze(1), reduction="sum")    # unsqueeze for BCE
                loss.backward()
                for (idx,theta) in enumerate(param_list):
                    Tconfs[k,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )    # compute Tconf
                    Tkins[k,idx] = torch.sum(momentum_list[idx]**2)                                # and Tkin
                    theta.grad.zero_()
                k += 1

            # update current params, forces, and momenta
            for j in range(0, len(param_list)):  
                param_list_curr[j] = param_list[j].detach().clone()
            forces_curr = copy.deepcopy(forces)
            momentum_list_curr = copy.deepcopy(momentum_list)
            U0 = U1.clone()

            rejected = False
            acc += 1

        else:    # reject sample

            if i % n_meas == 0:    # take measurement  
                Tconfs[k] = Tconfs[k-1]
                Tkins[k] = Tkins[k-1]
                k += 1

            rejected = True
        
        
        #### end MH Criterion ####
        
        # O Step
        with torch.no_grad():
            for p in momentum_list_curr:
                
                rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
                p.mul_(np.sqrt(a))
                p.add_(np.sqrt((1-a)*T) * rand)    # O step
        

        if i % 100000 == 0:
            print("Sampling iteration ",i," done! \n")


    #### end outer loop ####


    end_time = time.time()
    print("Sampling took {} seconds, i.e {} minutes."
          .format(end_time-start_time, (end_time-start_time)/60))  
    print("Acceptance rate: " + str(acc/N))
  

    # compute global Tconf and normalize 
    Tconfs[:,-1] = torch.sum(Tconfs, dim=1)
    Tkins[:,-1] = torch.sum(Tkins, dim=1)
    Nds[-1] = torch.sum(Nds)
    Tconfs /= Nds
    Tkins /= Nds

    return (Tconfs, Tkins)


### the versions below use reduction "mean" for loss computations -> different forces  ###


# def OBABO(model, param_list, criterion, dataloader, N, h, T, gamma, n_meas):
#     """OBABO"""

#     lam = 1e-3 # weight decay (i.e. prior force)

#     # turn off gradients for model parameters that are not being sampled. 
#     # IMPORTANT: all parameters not being sampled must lie in earlier layers than sampled params.
#     # otherwise, turning off gradients impedes proper gradient flow...
#     for theta in list(model.parameters()):
#         theta.requires_grad = False
#     for theta in param_list:
#         theta.requires_grad = True
#         if theta.grad is not None:
#             theta.grad.zero_()
    
#     # data iterators to get small and full batches
#     full_batch_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size = len(dataloader.dataset), shuffle=False)
#     full_batch_iter = iter(full_batch_loader) 
#     small_batch_iter = iter(dataloader)

#     # calculate initial force
#     (features, targets) = get_single_batch(dataloader, small_batch_iter)
#     loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#     loss.backward()

#     # build momentum and force buffers
#     momentum_list = []
#     forces = []
#     for theta in param_list:
#         dummy = theta.detach()
#         momentum_list.append(torch.zeros_like(dummy))
#         forces.append(-1*theta.grad - lam*dummy)
#         theta.grad.zero_() 

#     # get initial Tconfs/Tkins
#     Tconfs = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)   # stores Tconf for each sampled parameter, as well as global one
#     Tkins = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)    # stores Tkins in same manner
#     Nds = torch.zeros(len(param_list)+1)                                    # degrees of freedom, normalizes Tconfs/Tkins at the end
    
#     (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
#     loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#     loss.backward()
    
#     for (idx, theta) in enumerate(param_list):
#         Nds[idx] = theta.numel()     # count degs. of freedom
#         Tconfs[0,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
#         Tkins[0,idx] = torch.sum(momentum_list[idx]**2)                               # and Tkin
#         theta.grad.zero_()
    
#     k = 1                   # row index of Tconfs array to be filled     
    

#     # start sampling
#     a = np.exp(-gamma*h)    # OBABO constant
#     print("starting sampling...")
#     start_time = time.time()
#     for i in range(0,N):
        
#         # ABO steps
#         with torch.no_grad():
#             for (theta, p, force) in zip(param_list, momentum_list, forces):
#                 rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
#                 p.mul_(np.sqrt(a))
#                 p.add_(np.sqrt((1-a)*T) * rand + 0.5*h*force)     # O + B step
                
#                 theta.add_(h*p)    # A step
                   
#         # calculate forces
#         (features, targets) = get_single_batch(dataloader, small_batch_iter)
#         loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#         loss.backward()
#         for j in range(0, len(param_list)):
#             forces[j] = -1 * param_list[j].grad - lam*param_list[j].detach()
#             param_list[j].grad.zero_()  # prepare next backward() 
        
#         # OB steps
#         with torch.no_grad():
#             for (theta, p, force) in zip(param_list, momentum_list, forces):      
#                 p.add_(0.5*h*force)   # B step
#                 rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
#                 p.mul_(np.sqrt(a))
#                 p.add_(np.sqrt((1-a)*T) * rand)     # O step

#         # take measurement
#         if i % n_meas == 0:         
            
#             (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)  # get full gradients
#             loss = criterion(model(features), targets.unsqueeze(1)) 
#             loss.backward()
            
#             for (idx,theta) in enumerate(param_list):
#                 Tconfs[k,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
#                 Tkins[k,idx] = torch.sum(momentum_list[idx]**2)      # Tkin
#                 theta.grad.zero_()
#             k += 1
        
        
#         if i % 100000 == 0:
#             print("Sampling iteration ",i," done! \n")
            
        
#     end_time = time.time()
#     print("Sampling took {} seconds, i.e {} minutes."
#           .format(end_time-start_time, (end_time-start_time)/60))    
   
#     # compute global Tconf and normalize 
#     Tconfs[:,-1] = torch.sum(Tconfs, dim=1)
#     Tkins[:,-1] = torch.sum(Tkins, dim=1)
#     Nds[-1] = torch.sum(Nds)
#     Tconfs /= Nds
#     Tkins /= Nds
#     return (Tconfs, Tkins)
            



# def MOBABO(model, param_list, criterion, dataloader, N, h, T, gamma, L, SF_mode, n_meas):
#     """MOBABO"""

#     lam = 1e-3 # weight decay (i.e. prior force)

#     # turn off gradients for model parameters that are not being sampled. 
#     # IMPORTANT: all parameters not being sampled must lie in earlier layers than sampled params.
#     # otherwise, turning off gradients impedes proper gradient flow...
#     for theta in list(model.parameters()):
#         theta.requires_grad = False
#     for theta in param_list:
#         theta.requires_grad = True
#         if theta.grad is not None:
#             theta.grad.zero_()
    
    
#     # data iterators to get small and full batches
#     full_batch_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size = len(dataloader.dataset), shuffle=False)
#     full_batch_iter = iter(full_batch_loader) 
#     small_batch_iter = iter(dataloader)


#     # calculate initial force
#     (features, targets) = get_single_batch(dataloader, small_batch_iter)
#     loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#     loss.backward()


#     # build momentum and force buffers, as well as list holding "current" params
#     param_list_curr = []
#     momentum_list_curr = []
#     momentum_list = []
#     forces_curr = []
#     forces = []
#     for theta in param_list:
#         dummy = theta.detach()
#         param_list_curr.append(dummy.clone())                   # the current params don't need gradients. when they are copied into the network in case
#                                                                 # of rejection of the last sample, gradient tracking is activated by default.             
#         momentum_list_curr.append(torch.zeros_like(dummy))
#         forces_curr.append(-1*theta.grad - lam*dummy)
#         theta.grad.zero_() 
#     forces = copy.deepcopy(forces_curr)
#     momentum_list = copy.deepcopy(momentum_list_curr)

#     # get initial Tconfs/Tkins
#     Tconfs = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)   # stores Tconf for each sampled parameter, as well as global one
#     Tkins = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)    # stores Tkins in same manner
#     Nds = torch.zeros(len(param_list)+1)                                    # degrees of freedom, normalizes Tconfs/Tkins at the end
    
#     (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
#     loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#     loss.backward()
    
#     for (idx, theta) in enumerate(param_list):
#         Nds[idx] = theta.numel()     # count degs. of freedom
#         Tconfs[0,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
#         Tkins[0,idx] = torch.sum(momentum_list[idx]**2)                               # and Tkin
#         theta.grad.zero_()
    
#     k = 1          # row index of Tconfs/Tkins arrays to be filled     
    
    
#     # get initial potential energy
#     U0 = 0
#     with torch.no_grad():
#         for param0 in (param_list):
#             U0 += torch.sum(param0**2)
#     U0 = 1/2 * lam * U0 + loss
    
#     # start sampling
#     a = np.exp(-gamma*h)    # OBABO constant
#     rejected = False         # decides whether former proposal needs to be overwritten with current values
#     print("starting sampling...")
#     start_time = time.time()
    

#     #### Begin Sampling
#     acc = 0 # acceptance rate
#     for i in range(0,N):
        
#         # after rejection, reset proposed params to current params
#         if rejected == True:
#             with torch.no_grad():
#                 for idx in range(0, len(param_list_curr)):
#                     param_list[idx].copy_(param_list_curr[idx])

#             forces = copy.deepcopy(forces_curr)
#             if SF_mode:  # sign flip
#                 momentum_list_curr = [-1*p for p in momentum_list_curr]
#             momentum_list = copy.deepcopy(momentum_list_curr)
                
        
#         kin_energy = 0
#         U1 = 0
        
#         #### Begin Proposal
        
#         for j in range(0,L):
        
#             # ABO steps
#             with torch.no_grad():
#                 for (theta, p, force) in zip(param_list, momentum_list, forces):
#                     rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
#                     p.mul_(np.sqrt(a))
#                     p.add_(np.sqrt((1-a)*T) * rand)     # O step
                    
#                     kin_energy -= torch.sum(p**2)
                    
#                     p.add_(0.5*h*force)     # B step
                    
                    
#                     theta.add_(h*p)    # A step
                       
#             # calculate forces
#             (features, targets) = get_single_batch(dataloader, small_batch_iter)
#             loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#             loss.backward()

#             for m in range(0, len(param_list)):
#                 forces[m] = -1 * param_list[m].grad - lam*param_list[m].detach()
            
#             # OB steps
#             with torch.no_grad():
#                 for (theta, p, force) in zip(param_list, momentum_list, forces):      
#                     p.add_(0.5*h*force)   # B step
                    
#                     kin_energy += torch.sum(p**2)
                    
#                     rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
#                     p.mul_(np.sqrt(a))
#                     p.add_(np.sqrt((1-a)*T) * rand)     # O step
    
#                     theta.grad.zero_()  # prepare next backward() 
        
#         #### End Proposal

#         kin_energy *= 0.5

#         # compute U1
#         (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
#         with torch.no_grad():
#             for param in param_list:
#                 U1 += torch.sum(param**2)
#             loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#         U1 = 1/2 * lam * U1 + loss
        
#         #### MH Criterion
#         MH = torch.exp( (-1/T) * (U1 - U0 + kin_energy) )
#         if( torch.rand(1) < min(1., MH) ):                     # accept sample

#             if i % n_meas == 0:    # take measurement  
#                 (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)  # get full gradients
#                 loss = criterion(model(features), targets.unsqueeze(1)) 
#                 loss.backward()
#                 for (idx,theta) in enumerate(param_list):
#                     Tconfs[k,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
#                     Tkins[k,idx] = torch.sum(momentum_list[idx]**2)                               # and Tkin
#                     theta.grad.zero_()
#                 k += 1

#             # update current params, forces, and momenta
#             for j in range(0, len(param_list)):  
#                 param_list_curr[j] = param_list[j].detach().clone()
#             forces_curr = copy.deepcopy(forces)
#             momentum_list_curr = copy.deepcopy(momentum_list)
#             U0 = U1.clone()

#             rejected = False
#             acc += 1

#         else:                                               # reject sample

#             if i % n_meas == 0:    # take measurement  
#                 Tconfs[k] = Tconfs[k-1]
#                 Tkins[k] = Tkins[k-1]
#                 k += 1

#             rejected = True
        
#         #### End MH Criterion
        

#         if i % 100000 == 0:
#             print("Sampling iteration ",i," done! \n")

#     end_time = time.time()
#     print("Sampling took {} seconds, i.e {} minutes."
#           .format(end_time-start_time, (end_time-start_time)/60))
#     print("Acceptance rate: " + str(acc/N))

#     #### End Sampling
  

#     # compute global Tconf and normalize 
#     Tconfs[:,-1] = torch.sum(Tconfs, dim=1)
#     Tkins[:,-1] = torch.sum(Tkins, dim=1)
#     Nds[-1] = torch.sum(Nds)
#     Tconfs /= Nds
#     Tkins /= Nds

#     return (Tconfs, Tkins)




# def OMBABO(model, param_list, criterion, dataloader, N, h, T, gamma, L, SF_mode, n_meas):
#     """MOBABO"""

#     lam = 1e-3 # weight decay (i.e. prior force)

#     # turn off gradients for model parameters that are not being sampled. 
#     # IMPORTANT: all parameters not being sampled must lie in earlier layers than sampled params.
#     # otherwise, turning off gradients impedes proper gradient flow...
#     for theta in list(model.parameters()):
#         theta.requires_grad = False
#     for theta in param_list:
#         theta.requires_grad = True
#         if theta.grad is not None:
#             theta.grad.zero_()
    
    
#     # data iterators to get small and full batches
#     full_batch_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size = len(dataloader.dataset), shuffle=False)
#     full_batch_iter = iter(full_batch_loader) 
#     small_batch_iter = iter(dataloader)


#     # calculate initial force
#     (features, targets) = get_single_batch(dataloader, small_batch_iter)
#     loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#     loss.backward()


#     # build momentum and force buffers, as well as list holding "current" params
#     param_list_curr = []
#     momentum_list_curr = []
#     momentum_list = []
#     forces_curr = []
#     forces = []
#     for theta in param_list:
#         dummy = theta.detach()
#         param_list_curr.append(dummy.clone())                   # the current params don't need gradients. when they are copied into the network in case
#                                                                 # of rejection of the last sample, gradient tracking is activated by default.             
#         momentum_list_curr.append(torch.zeros_like(dummy))
#         forces_curr.append(-1*theta.grad - lam*dummy)
#         theta.grad.zero_() 
#     forces = copy.deepcopy(forces_curr)
#     momentum_list = copy.deepcopy(momentum_list_curr)

#     # get initial Tconfs/Tkins
#     Tconfs = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)   # stores Tconf for each sampled parameter, as well as global one
#     Tkins = torch.zeros((N//n_meas + 1, len(param_list)+1), dtype=float)    # stores Tkins in same manner
#     Nds = torch.zeros(len(param_list)+1)                                    # degrees of freedom, normalizes Tconfs/Tkins at the end
    
#     (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
#     loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#     loss.backward()
    
#     for (idx, theta) in enumerate(param_list):
#         Nds[idx] = theta.numel()     # count degs. of freedom
#         Tconfs[0,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
#         Tkins[0,idx] = torch.sum(momentum_list[idx]**2)                               # and Tkin
#         theta.grad.zero_()
    
#     k = 1          # row index of Tconfs array to be filled     
    
    
#     # get initial potential and kinetic energy
#     U0 = 0
#     K0 = 0
#     with torch.no_grad():
#         for (param0, p0) in zip(param_list, momentum_list):
#             U0 += torch.sum(param0**2)
#             K0 += torch.sum(p0**2)
#     U0 = 1/2 * lam * U0 + loss
#     K0 *= 1/2
    
#     # start sampling
#     a = np.exp(-gamma*h)    # OBABO constant
#     rejected = False         # decides whether former proposal needs to be overwritten with current values
#     print("starting sampling...")
#     start_time = time.time()
    

#     #### Begin Sampling
#     acc = 0 # acceptance rate
#     for i in range(0,N):
        
#          # O Step
#         with torch.no_grad():
#             for p in momentum_list_curr:
#                 rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
#                 p.mul_(np.sqrt(a))
#                 p.add_(np.sqrt((1-a)*T) * rand)     # O step
        
#         # reset proposed params to current params
#         if rejected == True:
#             with torch.no_grad():
#                 for idx in range(0, len(param_list_curr)):
#                     param_list[idx].copy_(param_list_curr[idx])

#             forces = copy.deepcopy(forces_curr)
#             if SF_mode:  # sign flip
#                 momentum_list_curr = [-1*p for p in momentum_list_curr]     # typically done within MH step,
#                                                                             # but should be fine here as neither SF nor
#                                                                             # O steps change invariant p-density.
        
#         momentum_list = copy.deepcopy(momentum_list_curr)
              

#         U1 = 0
#         K1 = 0
        
#         #### Begin Proposal
        
#         for j in range(0,L):
        
#             # AB steps
#             with torch.no_grad():
#                 for (theta, p, force) in zip(param_list, momentum_list, forces):                    
#                     p.add_(0.5*h*force)     # B step
#                     theta.add_(h*p)    # A step
                       
#             # calculate forces
#             (features, targets) = get_single_batch(dataloader, small_batch_iter)
#             loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#             loss.backward()

#             for m in range(0, len(param_list)):
#                 forces[m] = -1 * param_list[m].grad - lam*param_list[m].detach()
            
#             # B step
#             with torch.no_grad():
#                 for (theta, p, force) in zip(param_list, momentum_list, forces):      
#                     p.add_(0.5*h*force)   # B step
                    
#                     theta.grad.zero_()  # prepare next backward() 
        
#         #### End Proposal


#         # compute U1, K1
#         (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)
#         with torch.no_grad():
#             for (param, p) in zip(param_list, momentum_list):
#                 U1 += torch.sum(param**2)
#                 K1 += torch.sum(p**2)
#             loss = criterion(model(features), targets.unsqueeze(1))   # unsqueeze for BCE
#         U1 = 1/2 * lam * U1 + loss
#         K1 *= 1/2
        
        
#         #### MH Criterion
#         MH = torch.exp( (-1/T) * (U1 - U0 + K1 - K0) )
#         if( torch.rand(1) < min(1., MH) ):                     # accept sample

#             if i % n_meas == 0:    # take measurement  
#                 (features, targets) = get_single_batch(full_batch_loader, full_batch_iter)  # get full gradients
#                 loss = criterion(model(features), targets.unsqueeze(1)) 
#                 loss.backward()
#                 for (idx,theta) in enumerate(param_list):
#                     Tconfs[k,idx] = torch.sum( theta.detach()*(theta.grad+lam*theta.detach()) )   # compute Tconf
#                     Tkins[k,idx] = torch.sum(momentum_list[idx]**2)                               # and Tkin
#                     theta.grad.zero_()
#                 k += 1

#             # update current params, forces, and momenta
#             for j in range(0, len(param_list)):  
#                 param_list_curr[j] = param_list[j].detach().clone()
#             forces_curr = copy.deepcopy(forces)
#             momentum_list_curr = copy.deepcopy(momentum_list)
#             U0 = U1.clone()
#             K0 = K1.clone()

#             rejected = False
#             acc += 1

#         else:                                               # reject sample

#             if i % n_meas == 0:    # take measurement  
#                 Tconfs[k] = Tconfs[k-1]
#                 Tkins[k] = Tkins[k-1]
#                 k += 1

#             rejected = True
        
#         #### End MH Criterion
        
#         # O Step
#         with torch.no_grad():
#             for p in momentum_list_curr:
#                 rand = torch.normal(mean=torch.zeros(p.shape), std=torch.ones(p.shape))
#                 p.mul_(np.sqrt(a))
#                 p.add_(np.sqrt((1-a)*T) * rand)     # O step
        

#         if i % 100000 == 0:
#             print("Sampling iteration ",i," done! \n")

#     end_time = time.time()
#     print("Sampling took {} seconds, i.e {} minutes."
#           .format(end_time-start_time, (end_time-start_time)/60))  
#     print("Acceptance rate: " + str(acc/N))

#     #### End Sampling
  

#     # compute global Tconf and normalize 
#     Tconfs[:,-1] = torch.sum(Tconfs, dim=1)
#     Tkins[:,-1] = torch.sum(Tkins, dim=1)
#     Nds[-1] = torch.sum(Nds)
#     Tconfs /= Nds
#     Tkins /= Nds

#     return (Tconfs, Tkins)