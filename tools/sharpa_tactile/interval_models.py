"""One binary output per annotated interval, with isolated temporal state."""
import torch
from .models import FrameProbe


class IntervalProbe(FrameProbe):
    def __init__(self,input_kind,head_kind,hidden=128,layers=1):
        super().__init__(input_kind,head_kind,hidden,layers,num_classes=2)

    def forward(self,f6=None,deform=None,lengths=None):
        values=self.project(f6,deform)
        if lengths is None:
            lengths=torch.full((values.shape[0],),values.shape[1],dtype=torch.long)
        if self.head_kind=='mlp':
            mask=torch.arange(values.shape[1],device=values.device)[None,:]<lengths.to(values.device)[:,None]
            pooled=(values*mask[:,:,None]).sum(1)/lengths.to(values.device)[:,None]
            hidden=self.context(pooled)
        else:
            packed=torch.nn.utils.rnn.pack_padded_sequence(values,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,(h,_)=self.context(packed)
            hidden=h[-1]
        return self.prediction(hidden)


def interval_windows(raw,window=16):
    """Left edge padding from this interval only; never read a future tick."""
    import numpy as np
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=3 or raw.shape[1:]!=(5,6) or not len(raw):
        raise ValueError('Interval F6 must be nonempty [T,5,6]')
    padded=np.concatenate([np.repeat(raw[:1],window-1,axis=0),raw],axis=0)
    return np.stack([padded[t:t+window] for t in range(len(raw))])
