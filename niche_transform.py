import numpy
import cPickle

niches,names= cPickle.load(open("nodecalc/niche_calc.pkl","rb"))

rows={}
for k in range(niches.shape[0]):
 if niches[k].sum() == 1:
  rows[niches[k].argmax()] = k

def transform(invec):
 res=numpy.zeros((len(rows.keys())))
 for k in xrange(len(rows.keys())):
  res[k]=invec[rows[k]]
 return res
 
