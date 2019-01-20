import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')#GPU by default

from math import sqrt
import time

from torchvamp import vamp
from torchvamp import prior_gb
from torchvamp import lmmse_svd

M=2000
N=4000
A=torch.randn(M,N)/sqrt(N)
support=(torch.rand(N)<0.25)
w = torch.randn(N)*support.float()
y = A@w

t = time.time()
w_hat_rho = vamp(A, y, 1E-10, prior=prior_gb, prior_prmts=(0.25, 0, 1), true_coef=w, max_iter=250, verbose=1) 
elapsed = time.time() - t
print(elapsed)