#!/usr/bin/python3
import cPickle
import os
import sys
import time
import random as rnd
import numpy as np
import numpy
import random
import pickle as pickle
import MultiNEAT as NEAT
import copy
#from concurrent.futures import ProcessPoolExecutor, as_completed
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
    return behavior, (4 - error)**2,None

import networkx as nx

class novsearch:
  def __init__(self,g,params,evaluate,seed=1,checkpoint=False,checkpoint_interval=10,do_magic=True):
   #archive of past behaviors
   self.archive=[]
   #archive of the *genomes* that represent those past behaviors
   self.garchive=[]

   #evaluation function
   self.evaluate=evaluate

   #initialize the population
   self.pop = NEAT.Population(g, params, True, 1.0, seed)
   self.pop.RNG.Seed(seed)

   self.checkpoint=checkpoint
   self.ci = checkpoint_interval
   self.checkpt_counter=0
   self.do_magic=do_magic

  def do_gens(self,gens):
   evaluate=self.evaluate
   generations = 0
   pop=self.pop

   #do the requested number of generations of evolution
   for generation in range(gens):

        genome_list = NEAT.GetGenomeList(pop)
        fitness_list=[]
        behavior_list=[]

        #get the novb (behavior) for each genome in the pop
        for genome in genome_list:
         #evaluate genome in the domain to get behavior and fitness
         novb,fitness,extra = evaluate(genome) 

         if self.do_magic:
          #novb=np.sqrt(novb.mean(axis=0).flatten())
          novb=np.hstack([novb.max(axis=0),novb.min(axis=0)]).flatten()
          novb=novb/np.linalg.norm(novb)
          print novb.max(),novb.min()
          print novb.shape

         behavior_list.append(novb)

        #now calculate novelty scotres
        if True:
         print "calculating novelty..."
         behaviors = behavior_list
         fitness_list = []

         #judge the novelty of a new indiviudal by all the
         #behaviors of current population + archive
         compiled_array=numpy.array(self.archive+behavior_list)
         for k in behavior_list:
          fitness_list.append(calc_novelty(k,compiled_array))

         #randomly add one individual to archive per generation
         #you can do other things here... see original NS paper if interested..
         idx = random.randint(0,len(behaviors)-1)
         self.archive.append(behaviors[idx])
         self.garchive.append(NEAT.Genome(genome_list[idx]))

        #assign novelty as the fitness for each individual
        NEAT.ZipFitness(genome_list, fitness_list)
       
        if self.checkpoint and generation%self.ci==0:
         print "saving..."
         glist = [k for k in genome_list]
         #to_save = [self.garchive,self.archive,glist,behavior_list]
         to_save = [self.archive,self.garchive,glist,behavior_list]
         cPickle.dump(to_save,open("nov%d.pkl"%self.checkpt_counter,"wb"))
         self.checkpt_counter+=1       
         print "done!"
 
        """ 
        # test
        net = NEAT.NeuralNetwork()
        champ=pop.Species[0].GetLeader()
        champ.BuildPhenotype(net)
        #evaluate(champ,False)
        if vis: 
         img = np.zeros((500, 500, 3), dtype=np.uint8)
         img += 10
         NEAT.DrawPhenotype(img, (0, 0, 500, 500), net )
         cv2.imshow("nn_win", img)
         cv2.waitKey(1000)
        """

        print "before epoch"
        pop.Epoch()
        print "after epoch"
        generations = generation

#helper function to calculate the novelty of b given a list of behaviors beh
def calc_novelty(b,beh):
   b=numpy.array(b)

   beh=beh.copy()
   #calculate distance from b to all vectors in beh
   beh-=b
   beh*=beh
   beh=beh.sum(1)

   #sort distances, i.e. first entries will reflect distance to b's nearest neighbors
   beh.sort()

   #return the summed distance to 25 nearest neighbors (25 is a somewhat arbitrary parameter that you can change)
   return beh[:25].sum()+0.00001

class melites:
  def __init__(self,generator,params,seed_evals,evaluate,seed=1,checkpoint=False,checkpoint_interval=10000,history=False,optimize=False):
    self.do_history = history    
    self.history = nx.MultiDiGraph()
    self.generator = generator
    self.params = params
    self.evaluate = evaluate
    self.seed_evals = seed_evals
    self.checkpoint = checkpoint
    self.checkpoint_interval = checkpoint_interval
    self.optimize = optimize
    self.seed=seed
    g= generator()

    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    self.pop= pop
    self.species = pop.Species[0]

    _,beh,extra = evaluate(g)
    g.Destroy() 
   
    self.behavior_shape = beh.shape[0]
    self.reset_tries=10
    self.tries = numpy.ones(self.behavior_shape)*self.reset_tries
    self.elite_score = -numpy.ones(self.behavior_shape)
    self.elite_map = {}
    self.elite_extra = {}
    self.evals=0
    self.checkpt_counter=0
    self.greedy=optimize
    self.plots=[]

  def do_evals(self,num):
    r_indx = numpy.array(range(self.behavior_shape),dtype=int)
    for x in xrange(num):

     if x%10000==0:
      print 'eval %d' % x

     if self.checkpoint and ((self.evals+1)%self.checkpoint_interval==0):
      agg_data = []
      for k in range(self.elite_score.shape[0]):
       agg_data.append(self.elite_extra[k])
      agg_data = numpy.array(agg_data)
      self.plots.append(agg_data)
      
      cPickle.dump([self.elite_score,self.elite_map,self.evals,self.history,self.elite_extra,self.plots],open("fool%d-%d.pkl"%(self.seed,self.checkpt_counter),"wb"))
      #self.checkpt_counter+=1

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
     behavior=np.clip(behavior,0.0,1.0)
     to_update = np.nonzero(behavior>self.elite_score)[0]
     improve = np.any(behavior>(1.05*self.elite_score))

     if niche!=None and improve:
      self.tries[niche]=self.reset_tries

     nosave=True
     for idx in to_update:  

      baby= new_baby
      if not self.do_history and idx in self.elite_map:
       self.elite_map[idx].Destroy()
       baby= NEAT.Genome(new_baby)
      else:
       nosave=False

      old_score = self.elite_score[idx]
      self.elite_score[idx]=behavior[idx]
      
      self.elite_map[idx]=baby
      self.elite_extra[idx]=extra[:,idx]

      if self.do_history:
       if baby not in self.history:
        self.history.add_node(baby)

       if parent!=None:
        if parent not in self.history:
         self.history.add_node(parent)

        self.history.add_edge(parent,baby,source_niche=parent_niche,target_niche=idx,old_score=old_score,new_score=behavior[idx])

      if old_score*1.05 < behavior[idx]:
       self.tries[idx]=self.reset_tries

     if nosave:     
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
	params.PopulationSize = 500
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
 
        ns = novsearch(g,params,evaluate,10,do_magic=False)
        ns.do_gens(30)
        #print melites(generator,params,5000000,1000, evaluate,seed=1)
