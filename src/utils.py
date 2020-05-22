import numpy as np
import pandas as pd
def compute_generation(index, vg, vy, m, c, a, p, mu, ve):
    i = index
    vg[i] = (1-c)*vg[i-1] + m[i-1]/2 + (2*c-1)*m[i-2]/2
    vy[i] = ve + a@(vg[i] + 2*mu*(1+mu)*vg[i-1] + 2*mu*(1+mu)*m[i-1])@a.T
    vy[i] = vy[i].item()
    m[i] = (p/vy[i])*(vg[i] + mu*vg[i-1] + mu*m[i-1])@a.T@a@(vg[i] + mu*vg[i-1] + mu*m[i-1])


def simulate_generations(length, vg0, c, a, p, mu, ve):  
    vg = [0 for i in range(length+2)]
    vy = [0 for i in range(length+2)]
    m = [np.zeros(vg0.shape) for i in range(length+2)]
    vg[0] = vg0
    vg[1] = vg0
    #sure about that?
    for i in range(2, length+2):
        compute_generation(i, vg, vy, m, c, a, p, mu, ve)
    return vg[2:], vy[2:], m[2:]
