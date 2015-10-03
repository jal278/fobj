#!/usr/bin/python3
import numpy
import os
import sys
import time
import random as rnd
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT

import matplotlib
matplotlib.use('gtkagg')
import pylab as plt


from render_vox import render
from image_rec import run_image 

target_class = 682

sz_x = 10
sz_y = 20
sz_z = 10

coords = 5
coordinates = numpy.zeros((sz_x,sz_y,sz_z,coords))

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

coordinates=coordinates.reshape((sz_x*sz_y*sz_z,coords))

def evaluate(genome,debug=False,save=None):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    error = 0

    # do stuff and return the fitness
    tot_vox = sz_x*sz_y*sz_z
    voxels = numpy.zeros((tot_vox,4))
    for val in xrange(tot_vox):
     net.Flush()
     net.Input(coordinates[val]) #np.array([1., 0., 1.])) # can input numpy arrays, too
                                      # for some reason only np.float64 is supported
     for _ in range(3):
        net.Activate()
     o = net.Output()
     voxels[val,:]=o

    voxels = voxels.reshape((sz_x,sz_y,sz_z,4))
    thresh=0.5
    voxels[0,:,:,0]=thresh-0.01
    voxels[-1,:,:,0]=thresh-0.01

    voxels[:,0,:,0]=thresh-0.01
    voxels[:,-1,:,0]=thresh-0.01

    voxels[:,:,0,0]=thresh-0.01
    voxels[:,:,-1,0]=thresh-0.01

    img1 = render(voxels,45,0,save=save) 
    img2 = render(voxels,90,5) 
    img3 = render(voxels,135,0) 
    img4 = render(voxels,180,5) 
    img5 = render(voxels,225,0)
    imgs = [img1,img2,img3,img4,img5]
    #plt.imshow(img)
    #plt.show()
    results = run_image(imgs)  

    if debug:
     return imgs,results
    return float(results[:,target_class].prod()) #voxels.flatten().sum()
    
params = NEAT.Parameters()
params.PopulationSize = 100
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
params.MutateRemLinkProb = 0.02;
params.RecurrentProb = 0;
params.OverallMutationRate = 0.15;
params.MutateAddLinkProb = 0.08;
params.MutateAddNeuronProb = 0.01;
params.MutateWeightsProb = 0.90;
params.MaxWeight = 8.0;
params.WeightMutationMaxPower = 0.2;
params.WeightReplacementMaxPower = 1.0;

params.MutateActivationAProb = 0.0;
params.ActivationAMutationMaxPower = 0.5;
params.MinActivationA = 0.05;
params.MaxActivationA = 6.0;

params.MutateNeuronActivationTypeProb = 0.03;

params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 1.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;



plt.figure(figsize=(12,18))

def getbest(i):

    g = NEAT.Genome(0, 5, 0, 4, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, i)
    #pop.RNG.Seed(i)

    generations = 0
    for generation in range(250):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
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
    gen = getbest(run)
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
