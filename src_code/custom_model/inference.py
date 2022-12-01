import numpy as np
from custom_model.utils import *
from custom_model.losses import *
from custom_model.predictor import *
from keras.optimizers import Adam
from scipy.stats import beta
from scipy.stats import moment
from scipy import integrate
from scipy.special import psi
from scipy import special

def Entropy(inputs, para, epsilon=1e-4):
    resolution = 1/para['nb_classes']
    x = np.reshape(inputs,(-1,para['nb_classes']))
    x = x/np.sum(x,axis=-1)[:,np.newaxis]/resolution
    y = np.where(x>epsilon, -x*np.log(x)*resolution, 0)
    y = np.sum(y,axis=-1)
    return y.reshape(inputs.shape[:3])

def KLDivergence(x,inputs, para, epsilon=1e-4):
    resolution = 1/para['nb_classes']
    x = np.reshape(x,(x.shape[0], -1, para['nb_classes']))
    y = np.reshape(inputs,(-1, para['nb_classes']))
    
    z = []
    for i in range(len(x)):
        z_ = np.where(((y>epsilon)&(x[i]>epsilon)), x[i]*np.log(x[i]/y)*resolution, 0)
        z.append(np.sum(z_,axis=-1))
    z = np.stack(z,0)
    z = np.mean(z,0)
    
    return z.reshape(inputs.shape[:3])

def NLL_histogram(y_true, prediction):
    y_true = (y_true/2).astype(int).flatten()
    print(np.amax(y_true), np.amin(y_true))
    y_pred = prediction.reshape((-1,66))
    
    nll = np.zeros(len(y_true))
    for j in range(len(y_true)):
        nll[j] = y_pred[j,y_true[j]]
    
    nll = -np.log(nll*65)
    return nll

def OverallNLL(model, test_gen, nb_de, para):
    y_true = test_gen.y*130
    N = test_gen.__len__()
    nll_all = []
    var_a = []
    for s in range(N):
        var_epis, var_alea, entropy_epis, entropy_alea, prediction = single_inference(model, test_gen, sample_nb=s, nb_de=nb_de, para=para)
        if s!=N-1:
            nll = NLL_histogram(y_true[s*200:s*200+200], prediction)
            nll_all.append(nll)
        else:
            nll = NLL_histogram(y_true[s*200:], prediction)
            nll_all.append(nll)
        var_a.append(var_alea)
    nll_all = np.concatenate(nll_all)
    var_a = np.concatenate(var_a)
    return nll_all, var_a

def EnsembleInference(test_gen, para, nb_ensemble=10):
    model = build_crossmodel(para)
    model.compile(loss = nll_beta(), optimizer=Adam())
    
    y_true = test_gen.y*0.99*130
    At = []
    Bt = []
    
    for i in range(nb_ensemble):
        print(i, end='\r')
        model.load_weights('./DE/beta/model'+str(i)+'/model')
        
        y = model.predict(test_gen)
        w,k = np.split(y,2,axis=-1)
        w = w*0.98+0.01

        k = 1/(k**2+1e-4)+0.2
        a = w*k+1
        b = (1-w)*k+1
        
        At.append(a)
        Bt.append(b)
    
    At = np.stack(At,0)
    Bt = np.stack(Bt,0)
    
    mu = At/(At+Bt)*130
    var = At*Bt/(At+Bt)**2/(At+Bt+1)*130*130
    
    yp = np.mean(mu, 0)
    alea = np.mean(var, 0)
    epis = np.var(mu, 0)
    
    return At.squeeze(), Bt.squeeze(), yp.squeeze()#, alea.squeeze(), epis.squeeze()
        
def NLLEnsemble(a, b, y, nb_ensemble=10):
    prob = np.zeros_like(a)
    
    for i in range(nb_ensemble):
        p = beta.pdf(y/130, a[i], b[i], scale=1)
        prob[i] = p
        
    prob_m = np.mean(prob[:nb_ensemble], 0)
    print(-np.log(prob_m+1e-9).mean())
    return prob_m

def single_inference(model, test_gen, sample_nb, nb_de, para, sampling_mode='mean'):
    
    nb = para['nb_classes']
    mesh = np.arange(0, 130+65/(nb-1), 130/(nb-1))/130
    N = test_gen.__len__()
    assert sample_nb<N
    
    output_ensemble = []
    
    for i in range(1):
        model.load_weights('./DE/histogram/model'+str(i)+'/model')
        yp = model.predict(test_gen.__getitem__(sample_nb))
        yp = yp.transpose((3,0,1,2))/np.sum(yp,-1)
        yp = yp.transpose((1,2,3,0))
        output_ensemble.append(yp)
        
    output_ensemble = np.stack(output_ensemble, axis=0) #(D,B,T,N,nb)
    prediction = np.mean(output_ensemble, axis=0) #(B,T,N,nb)

    mu = np.mean(prediction*mesh, axis=-1) #(B,T,N)
    mu_d = np.mean(output_ensemble*mesh, axis=-1)
    
    var_epis = np.var(mu_d*130,axis=0)
    var_alea = np.mean(prediction*(np.tile(np.expand_dims(mu,3),(1,1,1,nb))-mesh)**2, axis=-1)
    
    entropy_alea = Entropy(prediction, para)
    entropy_epis = KLDivergence(output_ensemble, prediction, para)
    
    return var_epis, var_alea, entropy_epis, entropy_alea, prediction


def EntropyUncertainty(a, b):
    mesh = np.arange(0.5, 130.5, 1)/130
    
    Ea = np.zeros_like(a[0]) #(b, T, N)
    Ee = np.zeros_like(a[0])
    
    for i in range(a.shape[1]):
        print(i, end='\r')
        distributions = np.zeros((10, 10, 193, 130))
        for j in range(len(mesh)):
            distributions[...,j] = beta.pdf(mesh[j], a[:,i], b[:,i])
        dmu = np.mean(distributions, 0) # (T, N, 130)
        
        e_a = integrate.simpson(-dmu*np.log(dmu+1e-8), mesh)
        e_e = KLDivergence(distributions, dmu)
        Ea[i] = e_a
        Ee[i] = e_e
    return Ea, Ee
        
        
def UncertaintyEstimation(a, b):
    Ut = np.zeros_like(a[0])
    Ua = np.zeros_like(a[0])
    
    #entropy = np.log(special.beta(a,b)+1e-40)-(a-1)*psi(a)-(b-1)*psi(b)+(a+b-2)*psi(a+b)
    #Ua = np.mean(beta.entropy(a, b), 0)
    
    mesh = np.arange(0.5, 130.5, 1)/130
    
    for i in range(a.shape[1]):
        print(i, end='\r')
        distributions = np.zeros((10, 10, 193, 130))
        for j in range(len(mesh)):
            distributions[...,j] = beta.pdf(mesh[j], a[:,i], b[:,i])
        dmu = np.mean(distributions, 0) # (T, N, 130)
        
        Ut[i] = integrate.simpson(-dmu*np.log(dmu+1e-8), mesh)
        Ua[i] = np.mean(beta.entropy(a[:,i], b[:,i]), 0)
    return Ua, Ut-Ua    