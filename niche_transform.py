import numpy
import cPickle

niches,names= cPickle.load(open("nodecalc/niche_calc.pkl","rb"))

rows={}
include={}
reverse={}
for k in range(niches.shape[0]):
 if niches[k].sum() == 1:
  rows[niches[k].argmax()] = k
  include[k]=True
  reverse[k]=niches[k].argmax()

def transform3(invec):    
 res=numpy.zeros((invec.shape[0],len(rows.keys()),invec.shape[2]))
 for k in xrange(len(rows.keys())):
  res[:,k]=invec[:,rows[k]]
 return res

    
def transform2(invec):
 res=numpy.zeros((len(rows.keys()),invec.shape[1]))
 for k in xrange(len(rows.keys())):
  res[k]=invec[rows[k]]
 return res


def transform(invec):
 res=numpy.zeros((len(rows.keys())))
 for k in xrange(len(rows.keys())):
  res[k]=invec[rows[k]]
 return res
 
