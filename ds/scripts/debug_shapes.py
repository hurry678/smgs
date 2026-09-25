import numpy as np, torch, sys, pathlib
sys.path.insert(0,'/data2/hy/cts/e_problem/scripts')
from q2_models import Q2Model
z=np.load('/data2/hy/cts/e_problem/data/q2/q2_data.npz',allow_pickle=False)
a=torch.from_numpy(z['train_audio'][:4].astype(np.float32)); v=torch.from_numpy(z['train_vision'][:4].astype(np.float32)); t=torch.from_numpy(z['train_text_bert'][:4,1]>0)
obs=torch.stack([t,(a.abs().amax(-1)>1e-8),(v.abs().amax(-1)>1e-8)],1)
m=Q2Model('M0',pathlib.Path('/data2/hy/cts/e_problem/models/all-MiniLM-L6-v2'))
b,mm,tt=obs.shape
pieces=[]
for i in range(3):
 o=obs[:,i]
 local=torch.nn.functional.avg_pool1d(o.float().unsqueeze(1),5,1,2).squeeze(1)
 runs=m._run_stats(o); left,right=m._boundary_distance(o)
 max_run=torch.zeros((b,),device=obs.device); n_runs=torch.zeros((b,),device=obs.device)
 for bi in range(b):
  cur=mx=nr=0
  for j in range(tt):
   if bool(o[bi,j]):
    if cur: nr+=1; mx=max(mx,cur)
    cur=0
   else: cur+=1
  if cur: nr+=1; mx=max(mx,cur)
  max_run[bi]=mx; n_runs[bi]=nr
 max_run=max_run[:,None].expand(-1,tt)/tt; n_runs=n_runs[:,None].expand(-1,tt)/tt
 vals=[o.float(),local,runs/tt,left,right,o.float().mean(1,keepdim=True).expand(-1,tt).unsqueeze(-1),max_run.unsqueeze(-1),n_runs.unsqueeze(-1)]
 print('mod',i,[tuple(x.shape) for x in vals]); pieces.extend(vals)
nobs=obs.sum(1).float(); g=[(nobs/3.0).unsqueeze(-1),(nobs<=1.0).float().unsqueeze(-1)]
idx=(obs[:,0].long()+2*obs[:,1].long()+4*obs[:,2].long()); onehot=torch.nn.functional.one_hot(idx,8).float()[...,1:]; g.append(onehot)
print('global',[tuple(x.shape) for x in g]); pieces.extend(g)
print('all',[tuple(x.shape) for x in pieces]); print('cat',torch.cat(pieces,-1).shape)
