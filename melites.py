#!/usr/bin/python3
import cPickle
import os
import sys
import time
import random as rnd
import cv2
import numpy as np
import numpy
import random
import pickle as pickle
import MultiNEAT as NEAT
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def evaluate(genome):
    combos=[]
    base=range(4)
    cases=[0]*4

    for x in range(1,5):
     combos+=itertools.combinations(base,x)

    behavior=np.zeros(len(combos))

    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0

    # do stuff and return the fitness
    net.Flush()
    net.Input(np.array([1., 0., 1.])) # can input numpy arrays, too
                                      # for some reason only np.float64 is supported
    for _ in range(2):
        net.Activate()
    o = net.Output()

    case_error= abs(1 - o[0])
    error+=case_error
    cases[0] = 1.0-case_error

    net.Flush()
    net.Input([0, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()

    case_error= abs(1 - o[0])
    error +=case_error
    cases[1] = 1.0-case_error

    net.Flush()
    net.Input([1, 1, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    case_error= abs(o[0])
    error +=case_error
    cases[2] = 1.0-case_error

    net.Flush()
    net.Input([0, 0, 1])
    for _ in range(2):
        net.Activate()
    o = net.Output()
    case_error= abs(o[0])
    error +=case_error
    cases[3] = 1.0-case_error
   
    for x in range(len(combos)):
     t=0
     for element in combos[x]:
      t+=cases[element]
     behavior[x]=t
    return (4 - error)**2,behavior

def melites(generator,params, evals,seed_evals, evaluate,seed=1,checkpoint=False,checkpoint_interval=5000):
    g= generator()
    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)
    species = pop.Species[0]

    _,beh = evaluate(g)
    g.Destroy() 
   
    behavior_shape = beh.shape[0]

    elite_score = -numpy.ones(behavior_shape)
    elite_map = {}
    checkpt_counter=0
    for x in xrange(evals):

     if checkpoint and ((x+1)%checkpoint_interval==0):
      cPickle.dump([elite_score,elite_map],open("fool%d.pkl"%checkpt_counter,"wb"))
      checkpt_counter+=1
     if x<seed_evals:
      new_baby = generator()
     else:
      niche = random.randint(0,behavior_shape-1)
      new_baby = NEAT.Genome(elite_map[niche])
      species.MutateGenome(False,pop,new_baby,params,pop.RNG)

     _,behavior = evaluate(new_baby)
     to_update = np.nonzero(behavior>=elite_score)[0]

     for idx in to_update:  
      if idx in elite_map:
       elite_map[idx].Destroy()
      elite_score[idx]=behavior[idx]
      elite_map[idx]=NEAT.Genome(new_baby)
     
     new_baby.Destroy()     
    return elite_score,elite_map
    

def hillclimb(g,params,evals,evaluate,seed=1):
    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    species = pop.Species[0]
    champ = g
    c_fitness,beh = evaluate(champ)
    champ.SetFitness(c_fitness)
    champ.SetEvaluated()

    for x in xrange(evals):
     baby = NEAT.Genome(champ) #copy.copy(champ)
     species.MutateGenome(False,pop,baby,params,pop.RNG)
     b_fitness,beh = evaluate(baby)
     #print b_fitness, evaluate(champ)
     if b_fitness > c_fitness:
      #print b_fitness,evaluate(champ)
      c_fitness = b_fitness
      champ.Destroy() 
      champ = baby   
     else:
       baby.Destroy()
    return champ,c_fitness


if(__name__=='__main__'):
	params = NEAT.Parameters()
	params.PopulationSize = 150
	params.DynamicCompatibility = True
	params.WeightDiffCoeff = 4.0
	params.CompatTreshold = 2.0
	params.YoungAgeTreshold = 15
	params.SpeciesMaxStagnation = 15
	params.OldAgeTreshold = 35
	params.MinSpecies = 5
	params.MaxSpecies = 25
	params.RouletteWheelSelection = False
	params.RecurrentProb = 0.0
	params.OverallMutationRate = 0.8

	params.MutateWeightsProb = 0.90

	params.WeightMutationMaxPower = 2.5
	params.WeightReplacementMaxPower = 5.0
	params.MutateWeightsSevereProb = 0.5
	params.WeightMutationRate = 0.25

	params.MaxWeight = 8

	params.MutateAddNeuronProb = 0.03
	params.MutateAddLinkProb = 0.05
	params.MutateRemLinkProb = 0.0

	params.MinActivationA  = 4.9
	params.MaxActivationA  = 4.9

	params.ActivationFunction_SignedSigmoid_Prob = 0.0
	params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
	params.ActivationFunction_Tanh_Prob = 0.0
	params.ActivationFunction_SignedStep_Prob = 0.0

	params.CrossoverRate = 0.75  # mutate only 0.25
	params.MultipointCrossoverRate = 0.4
	params.SurvivalRate = 0.2

	generator = lambda :NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
        g=generator()
        #print hillclimb(g,params,10000,evaluate,10)
 
        print melites(generator,params,5000000,1000, evaluate,seed=1)
