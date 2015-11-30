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

    coords=np.zeros((10,3),dtype=np.float)
    net.Batch_input(coords,3)
    asfd

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
    return (4 - error)**2,behavior,None

import networkx as nx

class melites:
  def __init__(self,generator,params,seed_evals,evaluate,seed=1,checkpoint=False,checkpoint_interval=10000,history=False):
    self.do_history=history    
    self.history = nx.MultiDiGraph()
    self.generator=generator
    self.params= params
    self.evaluate = evaluate
    self.seed_evals = seed_evals
    self.checkpoint = checkpoint
    self.checkpoint_interval = checkpoint_interval

    g= generator()

    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    self.pop= pop
    self.species = pop.Species[0]

    _,beh,extra = evaluate(g)
    g.Destroy() 
   
    self.behavior_shape = beh.shape[0]
    self.reset_tries=5
    self.tries = numpy.ones(self.behavior_shape)*self.reset_tries
    self.elite_score = -numpy.ones(self.behavior_shape)
    self.elite_map = {}
    self.elite_extra = {}
    self.evals=0
    self.checkpt_counter=0
    self.greedy=True

  def do_evals(self,num):
    r_indx = numpy.array(range(self.behavior_shape),dtype=int)
    for x in xrange(num):

     if x%10000==0:
      print 'eval %d' % x

     if self.checkpoint and ((self.evals+1)%self.checkpoint_interval==0):
      cPickle.dump([self.elite_score,self.elite_map,self.evals,self.history],open("fool%d.pkl"%self.checkpt_counter,"wb"))
      self.checkpt_counter+=1

     parent=None
     parent_niche=None
     niche=None
     if x<self.seed_evals:
      new_baby = self.generator()
     else:
      if self.tries.sum()<=0:
       self.tries[:]=self.reset_tries

      print "pos p:", np.nonzero(self.tries>0)[0].shape    
      p=self.tries[r_indx]/self.tries.sum()
      niche = numpy.random.choice(r_indx,p=p) #random.randint(0,self.behavior_shape-1)

      if self.greedy:
        self.tries[niche]-=1

      parent=self.elite_map[niche]
      parent_niche=niche
      new_baby = NEAT.Genome(parent)
      self.species.MutateGenome(False,self.pop,new_baby,self.params,self.pop.RNG)

     _,behavior,extra = self.evaluate(new_baby)
     to_update = np.nonzero(behavior>self.elite_score)[0]
     improve = np.any(behavior>(1.05*self.elite_score))

     if niche!=None and improve:
      self.tries[niche]=self.reset_tries

     for idx in to_update:  

      if not self.do_history and idx in self.elite_map:
       self.elite_map[idx].Destroy()

      old_score = self.elite_score[idx]
      self.elite_score[idx]=behavior[idx]
      cloned_baby= NEAT.Genome(new_baby)
      self.elite_map[idx]=cloned_baby
      self.elite_extra[idx]=extra

      if self.do_history:
       if parent!=None:
        if parent not in self.history:
         self.history.add_node(parent)
        self.history.add_node(cloned_baby)
        self.history.add_edge(parent,cloned_baby,source_niche=parent_niche,target_niche=idx,old_score=old_score,new_score=behavior[idx])

      if old_score*1.05 < behavior[idx]:
       self.tries[idx]=self.reset_tries
     
     new_baby.Destroy()  
     self.evals+=1   

    return self.elite_score,self.elite_map,self.elite_extra

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
        def generator():
          return NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
         
	g=generator()
        #print hillclimb(g,params,10000,evaluate,10)
 
        print melites(generator,params,5000000,1000, evaluate,seed=1)
