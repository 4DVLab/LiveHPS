from Model.livehps import *
import torch
import numpy as np
from tqdm import tqdm

def load_GPUS(model,model_path,type=None,freze=False):
    state_dict = torch.load(model_path,map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if type==None:
            if 'sa2' in k.split('.'):
                continue
            if k[:6] == 'module':
                name = k[7:] # remove `module.
            else:
                name = k
        else:
            if k[:13] == f'module.model{type}':
                name = k[14:] # remove `module.`
            else:
                continue
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    if freze:
        for (n,p) in model.named_parameters():
            p.requires_grad=False
    return model

class ModelNet_shape(torch.nn.Module):
    def __init__(self,model1,model2,model3,model4,model5):
        super(ModelNet_shape,self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5

    def forward(self,seq_pc):
        seq_kp,fea = self.model1(seq_pc)
        B,T,kp_n = seq_kp.shape[0],seq_kp.shape[1],seq_kp.shape[2]
        kp_n = int(kp_n/3)
        new_kp,_,_ = self.model3(seq_kp,fea)
        trans = self.model2(seq_pc,new_kp.reshape(B,T,-1),fea.reshape(B,T,-1))
        rot,_,_ = self.model4(new_kp.reshape(B,T,-1),fea)
        shape,_,_ = self.model5(new_kp.reshape(B,T,-1),fea)

        return new_kp.reshape(B,T,-1),rot.reshape(B,T,-1),shape.reshape(B,-1),trans.reshape(B,T,1,3)
    
def LiveHPS():
    num_ms = 24
    model_kp1 = PBT(k=num_ms*3,channel=3)
    model1 = SMPL_trans(in_c=3+1024)
    model_kp2 = CPO(in_c=1024+72,hidden=256+3+1024,out_c=3,kp_num=num_ms)
    model_rot = CPO(in_c=1024+72,hidden=256+3+1024,out_c=6,kp_num=num_ms)
    model_shape = CPO(in_c=1024+72,hidden=256+3+1024,out_c=6,kp_num=num_ms,shape=True)
    model = ModelNet_shape(model_kp1,model1,model_kp2,model_rot,model_shape)
    load_GPUS(model,'./save_models/livehps.t7')
    return model
