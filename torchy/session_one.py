import numpy as np
import torch
import torch.nn as nn
from torch import tensor


def test(a, b, cmp, cname=None):
    if cname is None:
        cname = cmp.__name__
    assert cmp(a, b), f"{cname}:\n{a}\n{b}"

    
def test_eq(a, b):
    test(a, b, operator.eq, '==')

    
def near(a, b):
    return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a, b):
    test(a, b, near)
    

class Relu():
    def __call__(self, inp):  # forward, runs with Relu(in)
        self.inp = inp
        self.out = inp.clamp_min(0.) - 0.5
        return self.out
    
    def backward(self): 
        self.inp.g = (self.inp>0).float() * self.out.g

        
class Lin():
    def __init__(self, w, b): 
        self.w, self.b = w, b
        
    def __call__(self, inp):
        self.inp = inp
        self.out = inp @ self.w + self.b
        return self.out
    
    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        # Creating a giant outer product, just to sum it, is inefficient!
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
        self.b.g = self.out.g.sum(0)

            
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out
    
    def forward(self): 
        raise Exception('not implemented')

    def backward(self): 
        self.bwd(self.out, *self.args)
        

class Relu(Module):
    def forward(self, inp): 
        return inp.clamp_min(0.)-0.5
    
    def bwd(self, out, inp): 
        inp.g = (inp>0).float() * out.g
        
class Lin(Module):
    def __init__(self, w, b): 
        self.w,self.b = w,b
        
    def forward(self, inp): 
        return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = torch.einsum("bi,bj->ij", inp, out.g)
        self.b.g = out.g.sum(0)

        
class Mse(Module):
    def forward (self, inp, targ): 
        return (inp.squeeze() - targ).pow(2).mean()
    
    def bwd(self, out, inp, targ): 
        inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]

            
class Lin(Module):
    def __init__(self, w, b): 
        self.w,self.b = w,b
        
    def forward(self, inp): 
        return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum(0)

        
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        
    def __call__(self, x):
        for l in self.layers: 
            x = l(x)
        return x