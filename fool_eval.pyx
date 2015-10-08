import MultiNEAT as NEAT
from render_vox import render
import image_rec
from image_rec import run_image 
import numpy
from cpython cimport bool

sz_x = 20
sz_y = 20
sz_z = 20

coords = 6
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
   coordinates[_x,_y,_z,5]=x_grad[_x]**2+z_grad[_z]**2

coordinates=coordinates.reshape((sz_x*sz_y*sz_z,coords))

target_class = 681

def evaluate(genome,bool debug=False,save=None):
    verbose=True
    if verbose:
     print 'building...'

    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    genome.CalculateDepth()
    cdef int depth = genome.GetDepth()
    print depth
    error = 0

    # do stuff and return the fitness
    tot_vox = sz_x*sz_y*sz_z
    voxels = numpy.zeros((tot_vox,4))
    if verbose:
     print 'generating voxels...'


    cdef long int val
    for val in xrange(tot_vox):
     #net.Flush()
     net.Input(coordinates[val]) #np.array([1., 0., 1.])) # can input numpy arrays, too
                                      # for some reason only np.float64 is supported
     for _ in xrange(depth):
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

    if verbose:
     print 'rendering images'
    img1 = render(voxels,45,0,save=save) 
    img2 = render(voxels,90,5) 
    img3 = render(voxels,135,0) 
    img4 = render(voxels,180,5) 
    img5 = render(voxels,225,0)
    imgs = [img1,img2,img3,img4,img5]
    #plt.imshow(img)
    #plt.show()
    if verbose:
     print 'running image rec'
    results = run_image(imgs)  

    if debug:
     return imgs,results
    results = results.prod(axis=0)
    return float(results[target_class]),results #voxels.flatten().sum()
