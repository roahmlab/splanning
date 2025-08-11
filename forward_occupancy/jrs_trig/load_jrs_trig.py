"""
Load trigonometry version of precomtuted joint reacheable set (precomputed by CORA)
Author: Yongseok Kwon
Reference: Holmes, Patrick, et al. ARMTD
"""
import torch
from zonopy import gen_rotatotope_from_jrs_trig, gen_batch_rotatotope_from_jrs_trig
#from zonopy.transformations.homogeneous import gen_batch_H_from_jrs_trig
from zonopy import zonotope, polyZonotope, batchZonotope
from scipy.io import loadmat
import os


class JRSInfo:
    def __init__(self, jrs_path = None, dtype=torch.float32, device='cpu'):
        if jrs_path is None:
            dirname = os.path.dirname(__file__)
            jrs_path = os.path.join(dirname,'jrs_trig_tensor_saved/')

        self.cos_dim = 0 
        self.sin_dim = 1
        self.vel_dim = 2
        self.ka_dim = 3
        self.acc_dim = 3 
        self.kv_dim = 4
        self.time_dim = 5
        self.T_fail_safe = 0.5


        self.JRS_KEY = loadmat(jrs_path+'c_kvi.mat')
        self.g_ka = loadmat(jrs_path+'d_kai.mat')['d_kai'][0,0]
        #JRS_KEY = torch.tensor(JRS_KEY['c_kvi'],dtype=torch.float)

        '''
        qjrs_path = os.path.join(dirname,'qjrs_mat_saved/')
        qjrs_key = loadmat(qjrs_path+'c_kvi.mat')
        qjrs_key = torch.tensor(qjrs_key['c_kvi'])
        '''


        print("Loading JRS from", jrs_path)
        jrs_tensor = []
        for c_kv in self.JRS_KEY['c_kvi'][0]:
            jrs_filename = jrs_path+'jrs_trig_tensor_mat_'+format(c_kv,'.3f')+'.mat'
            jrs_tensor_load = loadmat(jrs_filename)
            jrs_tensor.append(jrs_tensor_load['JRS_tensor'].tolist()) 
        self.jrs_tensor = torch.tensor(jrs_tensor,dtype=dtype,device=device)