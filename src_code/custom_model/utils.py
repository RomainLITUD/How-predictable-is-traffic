import numpy as np
import h5py
from skimage.measure import block_reduce


def directed_adj():
    acc = [-1,36,47,58,74,99,110,144,154,192]
    
    base = np.identity(193,dtype=bool)
    
    for i in range(0,193):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][75]=True
    base[47][48]=True
    base[110][48]=True
    base[99][100]=True
    base[99][111]=True
    base[58][59]=True
    base[58][145]=True
    base[144][155]=True
    base[154][155]=True
    base[192][0]=True
    base[74][0]=True
    
    both = np.logical_or(base, base.transpose())
    
    return both.astype(int)

def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(x, mean, std):
    return x * std + mean

def DLdata(V, Q, obs, pred, interval=1, mode='multistep', time_period=0):
    V = np.transpose(V, (0,2,1))
    Q = np.transpose(Q, (0,2,1))
    D, T, N = V.shape
    V[V>129.] = 100.
    V = V/130.
    Q[Q>3000.] = 1000.
    Q = Q/3000.
    
    X = []
    Y = []
    Ind = []
    
    for i in range(D):
        for j in range(obs+1, T-15-1, interval):
            inp1 = V[i][j-obs:j]
            inp2 = Q[i][j-obs:j]
            inp = np.stack([inp1, inp2], axis=-1)
            if mode == 'singlestep':
                out = V[i][j+pred-1]
            else:
                out = V[i][j:j+pred]
            
            if np.amin(V[i][j-3:j+3]) < 1/3:
                X.append(inp)
                Y.append(out)
                Ind.append(np.array([time_period,j]))
    X0 = np.array(X)
    Y0 = np.array(Y)
    I0 = np.array(Ind)
    
    return X0, Y0, I0