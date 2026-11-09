"""Small analytic controls for the quantities used as experimental evidence."""
import torch
from probe import cosine,feasible,pair_rows

g={0:torch.tensor([1.,0.,0.]),2:torch.tensor([-1.,0.,0.]),
   6:torch.tensor([0.,1.,0.]),8:torch.tensor([1.,0.,0.])}
reg=torch.tensor([0.,10.,0.]);theta=torch.zeros(3)
rows=pair_rows(g,reg,theta,0)
feature=[row for row in rows if row['objective']=='feature']
assert len(feature)==6
assert sum(row['conflict'] for row in feature)==2
assert abs(feature[0]['cosine']+1)<1e-6
assert next(row for row in rows if row['objective']=='full')['cosine']>.9
assert torch.equal(feasible(torch.tensor([1.,-1.,-1.,1.]),torch.tensor([5.,-5.,5.,-5.])),
                   torch.tensor([0.,0.,-1.,1.]))
assert cosine(torch.zeros(3),torch.ones(3)) is None
print('ANALYTIC_METRIC_CONTROLS_PASSED')
