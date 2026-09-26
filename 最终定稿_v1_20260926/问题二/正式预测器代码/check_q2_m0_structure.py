import torch
from q2_models import Q2Model


def old_structure_stats(obs: torch.Tensor):
    b, m, t = obs.shape
    max_runs = torch.empty((b, m), device=obs.device, dtype=torch.float32)
    n_runs = torch.empty((b, m), device=obs.device, dtype=torch.float32)
    for bi in range(b):
        for mi in range(m):
            run = 0
            mr = 0
            nr = 0
            for ti in range(t):
                if obs[bi, mi, ti] < 0.5:
                    run += 1
                    mr = max(mr, run)
                    if run == 1:
                        nr += 1
                else:
                    run = 0
            max_runs[bi, mi] = mr
            n_runs[bi, mi] = nr
    return max_runs, n_runs


torch.manual_seed(20260924)
obj = Q2Model.__new__(Q2Model)
ok = True
for trial in range(20):
    obs = (torch.rand(8, 3, 64) < 0.42).float()
    runs = torch.stack([obj._run_stats(obs[:, i].bool()) for i in range(obs.shape[1])], dim=1)
    mx2 = runs.max(dim=2).values
    nr2 = ((runs == 1.0) & (runs > 0)).sum(dim=2).float()
    mx1, nr1 = old_structure_stats(obs)
    max_equal = torch.equal(mx1, mx2)
    nr_equal = torch.equal(nr1, nr2)
    if not (max_equal and nr_equal):
        print({"trial": trial, "max_equal": max_equal, "nr_equal": nr_equal})
        print("old_max", mx1)
        print("new_max", mx2)
        print("old_nr", nr1)
        print("new_nr", nr2)
        ok = False
        break

print({"all_equal": ok, "trials": 20 if ok else trial + 1})
if not ok:
    raise SystemExit(1)



