from OpenGL.GL import *
from OpenGL.GLU import *

from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GL.EXT.framebuffer_object import *

from ctypes import *

from math import *

import pygame
pygame.init()
pygame.display.set_mode((1,1), pygame.OPENGL|pygame.DOUBLEBUF)

glClearColor(0.0, 0.0, 0.0, 1.0)
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

glMatrixMode(GL_MODELVIEW);

done = False

t=0

while not done:

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
    gluLookAt(sin(t/200.0)*2,sin(t/500.0)*2,cos(t/200.0)*2,0,0,0,0,1,0)
    glMatrixMode(GL_MODELVIEW)

    # generate the texture we render to, and set parameters

    rendertarget=glGenTextures( 1 )

    glBindTexture( GL_TEXTURE_2D, rendertarget );
    glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                     GL_REPEAT);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                     GL_REPEAT );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    # occupy 512x512 texture memory

    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA,512,512,0,GL_RGBA,
                 GL_UNSIGNED_INT, None)

    # This is the interesting part: render-to-texture is initialized here

    # generate a "Framebuffer" object and bind it

    fbo=c_uint(1) # WTF? Did not find a way to get there easier
                  # A simple number would always result in a "Segmentation 
                  # Fault" for me
    glGenFramebuffers(1,fbo)

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);


    # render to the texture

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, 
                              GL_TEXTURE_2D, rendertarget, 0);

    # In case of errors or suspect behaviour, try this:
    #    print "Framebuffer Status:"
    #    print glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

    # Save Viewport configuration

    glPushAttrib(GL_VIEWPORT_BIT);

    # Align viewport to Texture dimensions - Try rendering 
    # to different dimensions than the texture has to find out 
    # about how your hardware handles display pitch

    glViewport(0, 0, 512, 512);

    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE);

    # Black background for the Helix

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    # Fallback to white

    glColor4f(1,1,1,1);

    # But try a fancy texture

    texture=glGenTextures( 1 )

    glBindTexture( GL_TEXTURE_2D, texture );
    glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                     GL_REPEAT);
    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                     GL_REPEAT );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

    texdata=[[[1.0,0,0,1],
              [1.0,1,1,1],
              [0.0,1,0,1],
              [1.0,1,1,1]],
             [[1.0,0,0,1],
              [1.0,0,0,1],
              [1.0,0,0,0.5],
              [0.0,0,0,1]],
             [[0.0,1,0,1],
              [0.0,0,0,0],
              [0.0,0,1,1],
              [0.0,0,0,0]],
             [[0.0,0,0,1],
              [0.0,0,0,1],
              [0.0,0,0,1],
              [0.0,0,0,1]]];

    glTexImage2Df(GL_TEXTURE_2D, 0,4,0,GL_RGBA,
                  texdata)

    glEnable( GL_TEXTURE_2D );
    glBindTexture( GL_TEXTURE_2D, texture );

    # The helix

    glBegin(GL_TRIANGLE_STRIP);

    for i in range(0,100):

        r=5.0;
        if (i%2==0):
            glTexCoord2f(0,i);
            glVertex3f( cos(i/r)*1.5, -2.5+i*0.05, sin(i/r)*1.5);
        else:
            glTexCoord2f(1,i);
            glVertex3f( cos(i/r+2)*1.5, -2.5+i*0.05, sin(i/r+2)*1.5);
        

    glEnd();
    glPopAttrib(); # Reset viewport to screen format
    # do not render to texture anymore - "switch off" rtt
    out = glReadPixels(0,0,512,512,GL_RGB,GL_FLOAT)
    from pylab import *
    imshow(out)
    show()

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); 
   


    # do not leak any mem

    glDeleteTextures(texture) 
    glDeleteTextures(rendertarget)
    glDeleteFramebuffers(1,fbo)
