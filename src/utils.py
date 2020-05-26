import numpy as np
import pandas as pd
def compute_generation(index, vg, vy, m, c, a, b, p, ve):
    i = index
    vg[i] = (1-c)*vg[i-1] + m[i-1]/2 + (2*c-1)*m[i-2]/2
    vy[i] = ve + a@vg[i]@a.T + 2*b@vg[i-1]@b.T + a@(m[i-1] + vg[i-1])@b.T + b@(m[i-1] + vg[i-1])@a.T + 2*b@m[i-1]@b.T
    vy[i] = vy[i].item()
    m[i] = (p/vy[i])*(vg[i]@a.T + (m[i-1] + vg[i-1])@b.T)@(vg[i]@a.T + (m[i-1] + vg[i-1])@b.T).T


def simulate_generations(length, vg0, c, a, b, p, ve):  
    vg = [0 for i in range(length+2)]
    vy = [0 for i in range(length+2)]
    m = [np.zeros(vg0.shape) for i in range(length+2)]
    vg[0] = vg0
    vg[1] = vg0
    #sure about that?
    for i in range(2, length+2):
        compute_generation(i, vg, vy, m, c, a, b, p, ve)
    h2 = [(a@vg[i]@a.T).item()/vy[i] for i in range(2,length+2)]
    return vg[2:], vy[2:], m[2:], h2
