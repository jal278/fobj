import cPickle
import numpy as np
import glob
from niche_transform import transform

#files = glob.glob("../Downloads/f1/*.pkl")
files = glob.glob("fool10-0.pkl")
#files = glob.glob("no-opt.pkl")
print files

avg=[]
for f in files:
 elite_score,elite_map,evals,history,elite_extra,plots = cPickle.load(open(f))

 if elite_score.shape[0]>1000:
  print "transform"
  elite_score = transform(elite_score)

 print elite_score.shape
 print elite_score.mean(),evals
 avg.append(elite_score.mean())

print np.mean(avg)
#f_new = f + ".new"
#cPickle.dump(stuff,open(f_new,"wb"),protocol=-1)
