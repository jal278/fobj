import pyximport; pyximport.install()

#!/usr/bin/python3
import numpy
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
   coordinates[_x,_y,_z,0]=1.0 #x_grad[_x]
   coordinates[_x,_y,_z,1]=x_grad[_x]
   coordinates[_x,_y,_z,2]=y_grad[_y]
   coordinates[_x,_y,_z,3]=z_grad[_z]
   coordinates[_x,_y,_z,4]=x_grad[_x]**2+y_grad[_y]**2+z_grad[_z]**2
   coordinates[_x,_y,_z,5]=x_grad[_x]**2+z_grad[_z]**2

coordinates=coordinates.reshape((sz_x*sz_y*sz_z,coords))

def evaluate(genome,debug=False,save=None):
    verbose=True
    if verbose:
     print 'building...'

    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    if verbose:
     print 'dcalc'
    #genome.CalculateDepth()
    if verbose:
     print 'dcalc complete'

    #depth = genome.GetDepth()
    depth=6

    error = 0

    # do stuff and return the fitness
    tot_vox = sz_x*sz_y*sz_z
    voxels = numpy.zeros((tot_vox,4))

    print "calling batch...", genome.NumNeurons()
    voxels = net.Batch_input(coordinates,depth)
    print "complete"
    """
    if verbose:
     print 'generating voxels...'
    for val in xrange(tot_vox):
     net.Flush()
     net.Input(coordinates[val]) #np.array([1., 0., 1.])) # can input numpy arrays, too
                                      # for some reason only np.float64 is supported
     for _ in xrange(depth):
        net.Activate()

     o = net.Output()
     voxels[val,:]=o
    """
    voxels = voxels.reshape((sz_x,sz_y,sz_z,4))
    thresh=0.5
    voxels[0,:,:,0]=thresh-0.01
    voxels[-1,:,:,0]=thresh-0.01

    voxels[:,0,:,0]=thresh-0.01
    voxels[:,-1,:,0]=thresh-0.01

    voxels[:,:,0,0]=thresh-0.01
    voxels[:,:,-1,0]=thresh-0.01

    bg_color = [net.neurons[k].time_const for k in range(3)]
    print bg_color 
    if verbose:
     print 'rendering images'
    angle_interval=45
    img1 = render(voxels,bg_color,0,0,save=save) 
    img2 = render(voxels,bg_color,45,5) 
    img3 = render(voxels,bg_color,90,0) 
    img4 = render(voxels,bg_color,135,5) 
    img5 = render(voxels,bg_color,180,0)
    img6 = render(voxels,bg_color,225,5)
    imgs = [img1,img2,img3,img4,img5,img6]

    #plt.imshow(img)
    #plt.show()
    if verbose:
     print 'running image rec'
    results = run_image(imgs)  

    if debug:
     return imgs,results
    results = results.prod(axis=0)
    return float(results[target_class]),results #voxels.flatten().sum()

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

if True:
 to_load = "fool100.pkl"
 stuff = cPickle.load(open(to_load,"rb"))
 plt.figure(figsize=(14,20))
 plt.ion()

 sort_list=zip(stuff[0],range(1000))
 sort_list.sort(reverse=True)
 for k in sort_list[:50]:
  print k,image_rec.labels[k[1]][:30]
 raw_input()
 for idx in range(0,1000):
  print image_rec.labels[idx][:50],stuff[0][idx]
  imgs,res = evaluate(stuff[1][idx],debug=True,save="out/out%d.ply"%idx) 
  plt.clf()

  fig = plt.gcf()
  fig.suptitle(image_rec.labels[idx][:30])
  subfig=1
  t_imgs = len(imgs)
  for img in imgs:
         plt.subplot(t_imgs,1,subfig)
         plt.title("Confidence: %0.2f%%" % (res[subfig-1,idx]*100.0))
         plt.imshow(img)
         subfig+=1
  plt.draw()
  plt.pause(0.1)
  plt.savefig("out/out%d.png"%idx)
 print "done.."
 asdf 

def generator(): 
     return NEAT.Genome(0, 6, 0, 4, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)

def mapelites(seed,evals,seed_evals,cpi):
    i = seed

    run = melites(generator,params,seed_evals,evaluate,checkpoint_interval=cpi,checkpoint=True)
    run.do_evals(evals) 

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

gens = []

for run in range(1):
    obj=False 
    if obj:
     gen = objective_driven(run)
    else:
     gen = mapelites(run,2000000,200,10000) #getbest(run)
    print('Run:', run, 'Generations to solve XOR:', gen)
    gens += [gen]

"""
with ProcessPoolExecutor(max_workers=8) as executor:
    fs = [executor.submit(getbest, x) for x in range(1000)]
    for i,f in enumerate(as_completed(fs)):
        gen = f.result()
        print('Run:', i, 'Generations to solve XOR:', gen)
        gens += [gen]

avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)

"""
