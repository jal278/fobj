import matplotlib
import saveply
matplotlib.use('tkagg')

from matplotlib.colors import hsv_to_rgb
#from colorsys import hsv_to_rgb

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *
from ctypes import *
from math import *

import mcubes

import numpy
import os
import sys
import time
import random as rnd
import cv2
import numpy as np
import pickle as pickle

import pygame
pygame.init()
pygame.display.set_mode((512,512), pygame.OPENGL|pygame.DOUBLEBUF)

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = numpy.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def render(voxels,angle1=45,angle2=10,save=None):
 sz_x,sz_y,sz_z,channels = voxels.shape
 thresh = 0.5
 #verts, faces = measure.marching_cubes(abs(voxels[:,:,:,0]), thresh)
 _verts,faces = mcubes.marching_cubes(voxels[:,:,:,0],thresh)

 glClearColor(0.0, 0.0, 0.0, 1.0)
 glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
 glMatrixMode(GL_MODELVIEW);

 if True:
    #print t

    # render a helix (this is similar to the previous example, but 
    # this time we'll render to a texture)

    # initialize projection

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION);    
    glLoadIdentity()

    gluPerspective(90,1,0.01,1000)
    gluLookAt(0,0,15, 0,0,0 ,0,1,0)

    glMatrixMode(GL_MODELVIEW)

    glShadeModel(GL_SMOOTH)

    glPushMatrix()
    glRotatef (angle1, 0.0, 1.0, 0.0); 
    glRotatef (angle2, 0.1, 0.0, 0.0); 

    #glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_LIGHTING)
    #glDisable(GL_CULL_FACE)
    #glDisable(GL_DEPTH_TEST)
    # Black background for the Helix
    glClearColor(0.5, 0.5, 0.5, 1.0)
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

    color_idx = np.asarray(_verts,dtype=int)


    colors = abs(voxels[color_idx[:,0],color_idx[:,1],color_idx[:,2],1:])
    colors = np.clip(colors,0,1)
    colors = hsv_to_rgb(colors)

    verts = _verts - numpy.array((sz_x/2,sz_y/2,sz_z/2))
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = verts[faces]
    tricols = colors[faces]

    if save!=None:
     saveply.save(save,verts,colors,faces)  
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = numpy.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    n=normalize_v3(n)
     
    #verts, faces = measure.marching_cubes(abs(voxels[:,:,:,0]), thresh)
    f_idx=0
    for tri in tris:
            #glNormal3f(*(n[f_idx]))
            colors=tricols[f_idx]
            f_idx+=1
            for k in xrange(3):
             #k.reverse()
             #print verts[k]
             #color = voxels[ints[0],ints[1],ints[2],1:] 
             #color = hsv_to_rgb(*color) #hsv_to_rgb(color)
             #print color
             glColor3f(*colors[k])
             glVertex3f( *tri[k])
             
    glEnd();
    glPopMatrix()
    # do not render to texture anymore - "switch off" rtt
    out = glReadPixels(0,0,512,512,GL_RGB,GL_FLOAT)
    return out

if (__name__=='__main__'):
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

  import pylab as plt
  plt.ion() 
  plt.show()
  ang=0
  while True:
   out = render(voxels,ang,ang)
   ang+=5
   plt.clf()
   plt.ion()
   plt.imshow(out)
   plt.pause(0.1) 
