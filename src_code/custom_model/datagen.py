from tensorflow.keras.utils import Sequence
import numpy as np
import h5py
from skimage.measure import block_reduce
from custom_model.utils import *
from keras.utils import to_categorical

np.random.seed(42)

class DataGenerator(Sequence):
    def __init__(self, para, batch_size, stage='train'):
        self.stage = stage
        self.batch_size = batch_size
        self.output_form = para['output_form']
        
        self.n_obs = para['obs']
        self.n_pred = para['pred']
        self.normalize = para['normalize']
        self.nb_classes = para['nb_classes']
        self.nb_nodes = para['nb_nodes']
        self.sigma = para['sigma']
        self.interval = para['interval']
        
        self.mesh = np.arange(0, 130+65/(self.nb_classes-1), 130/(self.nb_classes-1))
        
        Data = np.load('./data/'+stage+'.npz', allow_pickle=True)
        ve = np.array(Data['Ve'])
        qe = np.array(Data['Qe'])
        vm = np.array(Data['Vm'])
        qm = np.array(Data['Qm'])
        
        
        ve = block_reduce(ve,block_size=(1,1,2),func=np.mean)
        qe = block_reduce(qe,block_size=(1,1,2),func=np.mean)
        vm = block_reduce(vm,block_size=(1,1,2),func=np.mean)
        qm = block_reduce(qm,block_size=(1,1,2),func=np.mean)
        
        
        X1, Y1, I1 = DLdata(vm, qm, self.n_obs, self.n_pred, self.interval, 'multistep', 0)
        X2, Y2, I2 = DLdata(ve, qe, self.n_obs, self.n_pred, self.interval, 'multistep', 1)

        X = np.concatenate([X1, X2], axis=0)
        Y = np.concatenate([Y1, Y2], axis=0)
        I = np.concatenate([I1, I2], axis=0)

        if self.normalize:
            mean_v, mean_q, std_v, std_q = para['stats']
            X[...,0] = z_score(X[...,0], mean_v, std_v)
            X[...,1] = z_score(X[...,1], mean_q, std_q)
        if stage=='train':
            p = np.random.permutation(len(X))
            self.x = X[p]    
            self.y = Y[p]
            self.I = I[p]
        else:
            self.x = X    
            self.y = Y
            self.I = I

    def __len__(self,):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_i = self.I[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.stage=='train' or self.stage=='val':
            if self.output_form == 'beta':
                return batch_x, np.expand_dims(batch_y*0.99, 3)
            else:
                target = np.ones((self.nb_classes, len(batch_x), self.n_pred, self.nb_nodes))*batch_y*130
                target = target.transpose((1,2,3,0))
                target = np.exp(-((self.mesh-target)**2/self.sigma**2)/2)
                return batch_x, target
        else:
            if self.output_form == 'beta':
                return batch_x, np.expand_dims(batch_y*0.99, 3)
            else:
                return batch_x