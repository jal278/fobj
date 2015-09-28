import numpy
from plyfile import *

def save(name,verts,vertcols,faces):
 print "verts",verts.shape
 print "vertcols",vertcols.shape
 print "faces",faces.shape
 if faces.shape[0]==0:
  print "zero faces"
  return

 _verts = numpy.zeros((verts.shape[0],),dtype=[('x', 'f4'), ('y', 'f4'),
	('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
	('blue', 'u1')])
 _faces = numpy.zeros((faces.shape[0],),
	dtype=[('vertex_indices', 'i4', (3,))])

 _verts['x']=verts[:,0]
 _verts['y']=verts[:,1]
 _verts['z']=verts[:,2]
 _verts['red']=vertcols[:,0]*255
 _verts['green']=vertcols[:,1]*255
 _verts['blue']=vertcols[:,2]*255

 for idx in xrange(faces.shape[0]):
  _faces[idx] = faces[idx,:]

 el1 = PlyElement.describe(_verts, 'vertex')
 el2 = PlyElement.describe(_faces, 'face')
 PlyData([el1,el2],text="True").write(name)

if (__name__=='__main__'):
 vertex = numpy.array([(0, 0, 0,1,0,0),
	(0, 1, 1,0,255,0),
	(1, 0, 1,0,0,255),	
	(1, 1, 0,255,255,255)],
	dtype=[('x', 'f4'), ('y', 'f4'),
	('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
	('blue', 'u1')])
 face = numpy.array([([0, 1, 2],),
	([0, 2, 3],),
	([0, 1, 3],),
	([1, 2, 3],)],
	dtype=[('vertex_indices', 'i4', (3,))])

 el1 = PlyElement.describe(vertex, 'vertex')
 el2 = PlyElement.describe(face, 'face')

 PlyData([el1,el2],text="True").write('some_binary.ply')
