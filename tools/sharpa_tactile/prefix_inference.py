"""Streaming encoded features: predict final outcome without an interval end time."""
from pathlib import Path
import numpy as np
import torch
from .causal_prefix import PrefixProbe


class EncodedPrefixPredictor:
    def __init__(self,checkpoint,device='cpu'):
        self.device=torch.device(device)
        blob=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        assert blob['model_class']=='PrefixProbe'
        self.model=PrefixProbe(**blob['model_config']).to(self.device).eval()
        self.model.load_state_dict(blob['state_dict'],strict=True)
        self.reset()
    def reset(self):
        """Call at the start of each new interval; end time/outcome are not required."""
        self.state=None;self.positions_seen=0
    @torch.inference_mode()
    def step(self,f6=None,deform=None):
        values={}
        for key,raw,dim in [('f6',f6,1280),('deform',deform,2560)]:
            if key in self.model.input_kind:
                if raw is None:raise ValueError(f'Missing {key} feature')
                array=np.asarray(raw,dtype=np.float32)
                if array.shape!=(dim,) or not np.isfinite(array).all():raise ValueError(f'{key} must be finite [{dim}]')
                values[key]=torch.from_numpy(array).to(self.device)[None,None]
            else:values[key]=None
        logits,self.state=self.model(**values,state=self.state)
        self.positions_seen+=1
        return float(logits.sigmoid()[0,0])
