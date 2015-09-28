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

def evaluate(genome,debug=False):
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

    img1 = render(voxels,45,0) 
    img2 = render(voxels,90,10) 
    img3 = render(voxels,135,20) 
    img4 = render(voxels,180,30) 
    imgs = [img1,img2,img3,img4]
    #plt.imshow(img)
    #plt.show()
    results = run_image(imgs)  

    if debug:
     return imgs,results
    return float(results[:,680].sum()) #voxels.flatten().sum()
    
params = NEAT.Parameters()
params.PopulationSize = 40
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

params.ActivationFunction_SignedSigmoid_Prob = 1.0
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 1.0

params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2


def getbest(i):

    g = NEAT.Genome(0, 5, 0, 4, False, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID, 0, params)
    pop = NEAT.Population(g, params, True, 1.0, i)
    #pop.RNG.Seed(i)

    generations = 0
    for generation in range(50):
        genome_list = NEAT.GetGenomeList(pop)
        fitness_list = NEAT.EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
        NEAT.ZipFitness(genome_list, fitness_list)
        
        best = max([x.GetLeader().GetFitness() for x in pop.Species])
        print best
        imgs,res = evaluate(pop.Species[0].GetLeader(),debug=True)
        plt.ion()
        plt.clf()
        subfig=0
        t_imgs = len(imgs)
        for img in imgs:
         plt.subplot(t_imgs,1,subfig)
         plt.imshow(img)
         subfig+=1
        plt.draw()
        plt.pause(0.1)
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
