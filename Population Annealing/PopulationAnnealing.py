import numpy as np

"""
Package for population annealing

Basic structure:
    - an instance is primarily defined by its energy function and a solution region
    - from there, we have different attributes:
        * the cooling schedule
        * the population resampling method
        * the MCMC update rule, including how it generates moves, and how many steps to take for each beta
        * the number of replicas
        * error tolerance etc (stopping criteria)
    - what happens when we want to parallelise all of this?
"""


def energy(i):
    return np.cos(i - 10) - 0.2*i

def Resample(n_replicas,beta_current,beta_next,energies,current_posns):
    
    Boltzmann_factor = np.zeros((n_replicas,))
    for i in range(n_replicas):
        
        # Compute the statistical weight of the replica
        Boltzmann_factor[i] = np.exp(-energies[i]*(beta_next - beta_current))
    
    # Normalise the weights
    total = np.sum(Boltzmann_factor)
    weights = [i/total for i in Boltzmann_factor]
    
    # Compute the new number of replicas for each solution: here we just use the simplest implementation for now,
    # namely the multinomial distribution
    replicas_resampled = np.random.multinomial(n_replicas, weights)
    # Create the new list of positions
    new_posns = []
    for i in range(len(current_posns)):
        new_posns.extend([current_posns[i]]*replicas_resampled[i])
        
    return new_posns

def MCMC_Step(x,n_steps,n_replicas,posns,beta_current):
    
    updated_pos = np.zeros((n_replicas,))
    for i in range(n_steps):
        for j in range(n_replicas):

            # Simple update rule will be to take a random step of length < 1 from current position, left or right
            rand_step = 2*(np.random.rand() - 0.5)
            new_posn = posns[j] + rand_step
            
            current_energy = energy(posns[j])
            new_energy = energy(new_posn)

            # If the move takes us out of function range, stay where we are
            if new_posn > max(x) or new_posn < min(x):
                updated_pos[j] = posns[j]
            
            # Otherwise, accept or reject move?
            if new_energy < current_energy:
                updated_pos[j] = new_posn 
            else:
                p_accept = np.exp(-beta_current*(new_energy - current_energy))
                coin_flip = np.random.rand()
                if coin_flip < p_accept:
                    updated_pos[j] = new_posn
     
    return updated_pos

def Run_PopAnnealing(x,betas,MCsteps,replicas): 
    
    """
    other inputs needed: 
        energy function,
    """
    
    # Random intital solutions
    posns = np.random.choice(x,n_replicas)

    for beta in betas:
        
        # Do resampling
        
        for step in MCsteps:
            
            for replica in replicas:
                
                # Update positions according to Metropolis rule