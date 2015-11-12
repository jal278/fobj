import pyximport; pyximport.install()
import numpy
import math
import os
import sys
import time
import random as rnd
import numpy as np
import cPickle
import pickle as pickle
import MultiNEAT as NEAT

NEAT.import_array()

import matplotlib
matplotlib.use('gtkagg')
import pylab as plt

from render_vox_fast import render
import image_rec
from image_rec import run_image 
from melites import melites 
from fool_eval import evaluate

def load_niche_matrix(dummy=False):
 if dummy:
  niche_names = [k[:50] for k in image_rec.labels]
  niche_matrix = np.identity(1000)
  return niche_matrix,niche_names
 else:
  return cPickle.load(open("nodecalc/niche_calc.pkl","rb"))
  
niche_matrix,niche_names=load_niche_matrix()

target_class = 682
sz_x = 20
sz_y = 20
sz_z = 20

coords = 6
coordinates = numpy.zeros((sz_x,sz_y,sz_z,coords),dtype=np.double)

x_grad = numpy.linspace(-1,1,sz_x)
y_grad = numpy.linspace(-1,1,sz_y)
z_grad = numpy.linspace(-1,1,sz_z)

for _x in xrange(sz_x):
 for _y in xrange(sz_y):
  for _z in xrange(sz_z):
   coordinates[_x,_y,_z,0]=1.0 
   coordinates[_x,_y,_z,1]=x_grad[_x]
   coordinates[_x,_y,_z,2]=y_grad[_y]
   coordinates[_x,_y,_z,3]=z_grad[_z]
   coordinates[_x,_y,_z,4]=x_grad[_x]**2+y_grad[_y]**2+z_grad[_z]**2
   coordinates[_x,_y,_z,5]=x_grad[_x]**2+z_grad[_z]**2

coordinates=coordinates.reshape((sz_x*sz_y*sz_z,coords))

def evaluate(genome,debug=False,save=None):
    lighting=True
    verbose=True

    if verbose:
     print 'building...'

    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    if verbose:
     print 'dcalc'

    genome.CalculateDepth()
    if verbose:
     print 'dcalc complete'
    depth = genome.GetDepth()

    #fixed depth for now...
    #depth=6


    error = 0
    # do stuff and return the fitness
    tot_vox = sz_x*sz_y*sz_z
    voxels = numpy.zeros((tot_vox,4))

    print "calling batch...", genome.NumNeurons()
    voxels = net.Batch_input(coordinates,depth)
    print "complete"

    voxels = voxels.reshape((sz_x,sz_y,sz_z,4))
    thresh=0.5
    voxels[0,:,:,0]=thresh-0.01
    voxels[-1,:,:,0]=thresh-0.01

    voxels[:,0,:,0]=thresh-0.01
    voxels[:,-1,:,0]=thresh-0.01

    voxels[:,:,0,0]=thresh-0.01
    voxels[:,:,-1,0]=thresh-0.01

    bg_color = [net.neurons[k].time_const for k in range(3)]
    oparam = np.clip([net.neurons[k].bias for k in range(3)],0,1)
    print bg_color 
    print oparam

    if verbose:
     print 'rendering images'

    theta=45
    jitter=5

    shiny = oparam[0]*128
    spec = oparam[1]
    amb = oparam[2]

    img1 = render(voxels,bg_color,0,0,save=save,shiny=shiny,spec=spec,amb=amb,lighting=lighting) 
    img2 = render(voxels,bg_color,theta,jitter,shiny=shiny,spec=spec,amb=amb,lighting=lighting) 
    img3 = render(voxels,bg_color,theta*2,0,shiny=shiny,spec=spec,amb=amb,lighting=lighting) 
    img4 = render(voxels,bg_color,theta*3,jitter,shiny=shiny,spec=spec,amb=amb,lighting=lighting) 
    img5 = render(voxels,bg_color,theta*4,0,shiny=shiny,spec=spec,amb=amb,lighting=lighting)
    img6 = render(voxels,bg_color,theta*5,jitter,spec=spec,amb=amb,lighting=lighting)

    imgs = [img1,img2,img3,img4,img5,img6]

    if verbose:
     print 'running image rec'

    results = run_image(imgs)  

    if debug:
     return imgs,results

    results = results.prod(axis=0)
    
    niche_computation = np.dot(results,niche_matrix.T)
    print niche_computation.shape

    return float(results[target_class]),results 

#NEAT setup
params = NEAT.Parameters()
params.PopulationSize = 50
params.DynamicCompatibility = True
params.WeightDiffCoeff = 4.0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 15
params.RouletteWheelSelection = False
params.RecurrentProb = 0.0
params.MutateRemLinkProb = 0.0;
params.OverallMutationRate = 0.15;
params.MutateAddLinkProb = 0.13;
params.MutateAddNeuronProb = 0.03;
params.MutateWeightsProb = 0.90;
params.MaxWeight = 6.0;
params.WeightMutationMaxPower = 0.2;
params.WeightReplacementMaxPower = 1.0;

params.TimeConstantMutationMaxPower=0.2
params.BiasMutationMaxPower=0.2
params.MutateNeuronBiasesProb = 0.05
params.MutateNeuronTimeConstantsProb = 0.05
params.MaxNeuronTimeConstant =1.0
params.MinNeuronTimeConstant =0.0
params.MaxNeuronBias = 1.0
params.MinNeuronBias = 0.0

params.MutateActivationAProb = 0.05;
params.ActivationAMutationMaxPower = 0.5;
params.MinActivationA = 0.05;
params.MaxActivationA = 6.0;

params.MutateNeuronActivationTypeProb = 0.03;

params.ActivationFunction_SignedSigmoid_Prob = 1.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 0.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 0.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 1.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;

 
if False:
 #to_load = "fool100.pkl"
 to_load = "fool45.pkl"
 stuff = cPickle.load(open(to_load,"rb"))
 plt.figure(figsize=(16,22))
 plt.ion()

 num_niches= len(niche_names)
 sort_list=zip(stuff[0],range(num_niches))
 sort_list.sort(reverse=True)

 for k in sort_list[:50]:
  print k,niche_names[k[1]]

 raw_input()

 for _idx in range(0,num_niches):
  idx = sort_list[_idx][1]
  print niche_names[idx],stuff[0][idx]
  imgs,res = evaluate(stuff[1][idx],debug=True,save="out/out%d.ply"%_idx) 
  plt.clf()

  fig = plt.gcf()
  fig.suptitle(niche_names[idx][:30])
  subfig=1
  t_imgsx = (math.ceil(float(len(imgs)/3)))
  t_imgsy = 3
  
  for img in imgs:
         plt.subplot(t_imgsx,t_imgsy,subfig)
         plt.title("Confidence: %0.2f%%" % (res[subfig-1,idx]*100.0))
         plt.imshow(img)
         subfig+=1
  plt.draw()
  plt.pause(0.1)
  plt.savefig("out/out%d.png"%_idx)
 print "done.."
 exit()
rng=NEAT.RNG()

#genome generator
def generator(): 
     global rng
     g= NEAT.Genome(0, 6, 0, 4, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
     g.RandomizeParameters(rng)
     return g

#wrapper to call map elites
def mapelites(seed,evals,seed_evals,cpi):
    global rng
    rng.Seed(seed)
 
    run = melites(generator,params,seed_evals,evaluate,checkpoint_interval=cpi,checkpoint=True,seed=seed)
    run.do_evals(evals) 

#if you want to do objective-driven search
def objective_driven(seed):
    i = seed
    g = NEAT.Genome(0, 6, 0, 4, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, i)
    #pop.RNG.Seed(i)

    generations = 0
    for generation in range(250):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        fitness_list = [k[0] for k in fitness_list]
        NEAT.ZipFitness(genome_list, fitness_list)
        
        best_fits = [x.GetLeader().GetFitness() for x in pop.Species]
        best = max(best_fits)
        idx = best_fits.index(best)
        print best,pop.Species[idx].GetLeader().GetFitness()
        imgs,res = evaluate(pop.Species[idx].GetLeader(),debug=True,save="gen%d.ply"%generation)

        plt.ion()
        plt.clf()
        subfig=1
        t_imgs = len(imgs)
        for img in imgs:
         plt.subplot(t_imgs,1,subfig)
         plt.title("Confidence: %0.2f%%" % (res[subfig-1,target_class]*100.0))
         plt.imshow(img)
         subfig+=1
        plt.draw()
        plt.pause(0.1)
        plt.savefig("out%d.png"%generation)
        pop.Epoch()

        generations = generation

    return generations

obj=False
seed=10

if __name__=='__main__':
    obj=False 
    if obj:
     gen = objective_driven(seed)
    else:
     gen = mapelites(seed,2000000,200,10000) #getbest(run)

