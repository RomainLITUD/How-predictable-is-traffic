import numpy as np
from custom_model.utils import *
from custom_model.losses import *
from custom_model.predictor import *
from custom_model.datagen import *
from keras.optimizers import Adam
from scipy.stats import beta
from scipy.stats import moment
from scipy import integrate
from scipy.special import psi
from scipy import special
from scipy.integrate import simpson


def EnsemblePrediction(model, para, stage, nb_de, batch_size=100, mode=['prediction'], sampling_mode='mean'):
    assert stage in ['testin', 'testout']
    test_gen = DataGenerator(para, batch_size, stage)
    
    y_true = test_gen.y*130
    
    nb = para['nb_classes']
    mesh = np.arange(0, 130+65/(nb-1), 130/(nb-1))
    N = test_gen.__len__()
    
    Distributions = []
    Predictions = []
    Var_a = []
    Var_e = []
    Entropy_a = []
    Entropy_e = []
    
    for i in range(N):
        dis_sub = []
        xin = test_gen.__getitem__(i)
        for j in range(nb_de):
            print(i*batch_size, j+1, end='\r')
            model.load_weights('./DE/histogramf/model'+str(j)+'/model')
            yp = model.predict(xin,verbose=0)
            yp = np.transpose(yp, (3,0,1,2))/np.sum(yp, -1)
            yp = np.transpose(yp, (1,2,3,0))
            dis_sub.append(yp)
        # add analysis here
        dis_sub = np.array(dis_sub) #(10, T, N, nb)
        
        d_avr = np.mean(dis_sub, axis=0)
        
        if 'distribution' in mode:
            Distributions.append(d_avr)
        if 'prediction' in mode:
            pred = PredictionSampling(d_avr, mesh, sampling_mode)
            Predictions.append(pred)           
        if 'uncertainty' in mode:
            vara, vare = VarianceUncertainty(dis_sub, mesh)
            ua, ue = EntropyUncertainty(dis_sub, d_avr)
            
            Var_a.append(vara)
            Var_e.append(vare)
            Entropy_a.append(ua)
            Entropy_e.append(ue)
    
    if 'distribution' in mode:
        Distributions =  np.concatenate(Distributions, axis=0)
    if 'prediction' in mode:
        Predictions =  np.concatenate(Predictions, axis=0)
    if 'uncertainty' in mode:
        Var_a =  np.concatenate(Var_a, axis=0)
        Var_e =  np.concatenate(Var_e, axis=0)
        Entropy_a =  np.concatenate(Entropy_a, axis=0)
        Entropy_e =  np.concatenate(Entropy_e, axis=0)
        
    return Distributions, Predictions, Var_a, Var_e, Entropy_a, Entropy_e, y_true


def PredictionSampling(D, mesh, sampling_mode, dist=20):
    if sampling_mode=='mean':
        return simpson(D*mesh, mesh)
    if sampling_mode=='unimodal':
        return np.argmax(D, -1)
    if sampling_mode=='bimodal':
        yp1 = np.argmax(D,-1)
        yp2 = np.zeros_like(yp1)
        b, t, n = yp1.shape[0], yp1.shape[1], yp1.shape[2]
        for i in range(b):
            for j in range(t):
                for k in range(n):
                    dr = D[i,j,k].copy()
                    ind = yp1[i,j,k]
                    dr[max(ind-dist, 0):ind+dist+1]=0
                    yp2[i,j,k] = np.argmax(dr)
        return np.stack([yp1, yp2], -1)
        
    
def VarianceUncertainty(d_all, mesh):
    mu_all = simpson(d_all*mesh, mesh)
    
    mesh_r = mesh.reshape((1,1,1,1,131))
    d_ref = np.tile(mesh_r, (d_all.shape[0], d_all.shape[1],d_all.shape[2], d_all.shape[3], 1))
    
    d_ref = np.transpose(d_ref, (4,0,1,2,3))-mu_all
    d_ref = np.transpose(d_ref, (1,2,3,4,0))**2
    
    var_all = simpson(d_ref*d_all, mesh)
    
    vara = np.mean(var_all, axis=0)
    vare = np.var(mu_all, 0)
    
    return vara, vare

def EntropyUncertainty(dis_sub, d_avr):
    Ua = Entropy(d_avr)
    Ue = KLDivergence(dis_sub, d_avr)
    
    return Ua, Ue
    

def Entropy(inputs, epsilon=1e-4):
    x = np.reshape(inputs,(-1,131))
    x = x/np.sum(x,axis=-1)[:,np.newaxis]
    y = np.where(x>epsilon, -x*np.log(x), 0)
    y = np.sum(y,axis=-1)
    return y.reshape(inputs.shape[:3])

def KLDivergence(x,inputs, epsilon=1e-4):
    x = np.reshape(x,(x.shape[0], -1, 131))
    y = np.reshape(inputs,(-1, 131))
    
    z = []
    for i in range(len(x)):
        z_ = np.where(((y>epsilon)&(x[i]>epsilon)), x[i]*np.log(x[i]/y), 0)
        z.append(np.sum(z_,axis=-1))
    z = np.stack(z,0)
    z = np.mean(z,0)
    
    return z.reshape(inputs.shape[:3])

def Calibration(D, yt):
    b, t, n = D.shape[0], D.shape[1], D.shape[2]
    ppf = np.zeros_like(yt)
    for i in range(b):
        print(i, end='\r')
        for j in range(t):
            for k in range(n):
                ppf[i,j,k] = np.sum(D[i,j,k][:int(yt[i,j,k])+1])
    cali = np.zeros(51)
    for i in range(51):
        mask = (ppf<i*0.02)
        cali[i] = np.count_nonzero(mask)/b/t/n
    return cali, ppf