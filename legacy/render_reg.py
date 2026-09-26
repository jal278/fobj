import matplotlib
matplotlib.use('gtkagg')

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *
from ctypes import *
from math import *

from mayavi import mlab # doctest: +SKIP
from skimage import measure
import mcubes

import numpy
import os
import sys
import time
import random as rnd
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
import pylab as plt
import random

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

tot_vox = sz_x*sz_y*sz_z
voxels = numpy.zeros((tot_vox,4))
for val in xrange(tot_vox):
     voxels[val,0] = sum( (coordinates[val,1:])**2 )
     voxels[val,1] = ((1.0 - coordinates[val,1])+(coordinates[val,2]**2))/2.0 #np.random.random((3))
voxels = voxels.reshape((sz_x,sz_y,sz_z,4))
thresh = 0.5
#verts, faces = measure.marching_cubes(abs(voxels[:,:,:,0]), thresh)
_verts,faces = mcubes.marching_cubes(voxels[:,:,:,0],thresh)

"""
mlab.triangular_mesh([vert[0] for vert in verts],
      [vert[1] for vert in verts],
      [vert[2] for vert in verts],
      faces) # doctest: +SKIP
mlab.show() # doctest: +SKIP
"""

#import pygame
#pygame.init()
#pygame.display.set_mode((1,1), pygame.OPENGL|pygame.DOUBLEBUF)

from OpenGL.GL import *
from OpenGL.GLU import *

from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *

from ctypes import *

from math import *

import pygame
pygame.init()
pygame.display.set_mode((512,512), pygame.OPENGL|pygame.DOUBLEBUF)

glClearColor(0.0, 0.0, 0.0, 1.0)
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
glMatrixMode(GL_MODELVIEW);

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = numpy.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

plt.ion()
plt.show()
t=0
while True:
    print t
    # step time

    t=t+1

    # render a helix (this is similar to the previous example, but 
    # this time we'll render to a texture)

    # initialize projection

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION);    
    glLoadIdentity()

    gluPerspective(90,1,0.01,1000)
    gluLookAt(0,0,10, 0,0,0 ,0,1,0)

    glMatrixMode(GL_MODELVIEW)

    glShadeModel(GL_SMOOTH)

    glPushMatrix()
    spin = t*5
    glRotatef (spin, 0.0, 1.0, 0.0); 

    #glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_LIGHTING)
    #glDisable(GL_CULL_FACE)
    #glDisable(GL_DEPTH_TEST)
    # Black background for the Helix
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    # Fallback to white


    #lightZeroPosition = [0.,50.,-2.,1.]
    #lightZeroColor = [1.8,1.0,0.8,1.0] #green tinged
    #glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
    #glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
    #glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    #glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    #glEnable(GL_LIGHT0)
    # The helix

    #color = [1.0,0.,0.,1.]
    #glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
    glBegin(GL_TRIANGLES);
    verts = _verts - numpy.array((sz_x/2,sz_y/2,sz_z/2))
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = verts[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = numpy.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    n=normalize_v3(n)
    glColor3f(0.5,0.5,0.5);

    #verts, faces = measure.marching_cubes(abs(voxels[:,:,:,0]), thresh)
    f_idx=0
    for tri in tris:
            #glNormal3f(*(n[f_idx]))
            f_idx+=1
            for k in tri:
             k=list(k)
             #k.reverse()
             ints = map(int,k)

             #print verts[k]
             color = voxels[ints[0],ints[1],ints[2],1:] 
             color = list(color)
             #print color
             glColor3f(*color);
             glVertex3f( *k)
             
    glEnd();
    glPopMatrix()
    # do not render to texture anymore - "switch off" rtt
    out = glReadPixels(0,0,512,512,GL_RGB,GL_FLOAT)
    plt.clf()
    plt.imshow(out)
    plt.draw()
    plt.pause(0.1)

