import torch
p='/data2/hy/hy/models/modelscope_cache/all-MiniLM-L6-v2/pytorch_model.bin'
d=torch.load(p,map_location='cpu')
print(type(d),len(d))
for k,v in d.items(): print(k,tuple(v.shape))
