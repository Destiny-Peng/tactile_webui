"""Last-position one-sided Gaussian soft outcomes; independent of window overlap."""
import numpy as np

SIGMAS=(1,2,3,4,6,8)
CLASSES=('in_progress','success','failure')


def one_sided_targets(positions,key_position,outcome,sigma):
    if sigma<=0 or outcome not in (1,2):raise ValueError('positive sigma; success=1/failure=2')
    distance=np.maximum(float(key_position)-np.asarray(positions,dtype=np.float64),0)
    confidence=np.exp(-.5*(distance/float(sigma))**2)
    targets=np.zeros((len(distance),3),dtype=np.float32)
    targets[:,0]=1-confidence;targets[:,outcome]=confidence
    return targets


def fixed_reference(positions,key_position,outcome):
    return np.where(np.asarray(positions)<key_position,0,outcome).astype(np.int64)
