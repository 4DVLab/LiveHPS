import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder
from torch.autograd import Variable
from .transformer_d import TransformerDecoderLayer
from .pos_encoding import PositionalEncoding

class PBT(nn.Module):
    def __init__(self,k=36, channel=3,f_only=False):
        super(PBT, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.f_only= f_only
        if not f_only:
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, k)
        self.gru = nn.GRU(1024,512,bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(p=0.0)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def encode(self, x):
        x,_,_ = self.feat(x)
        return x
    
    def forward(self,x):
        xyz = x.permute(0,1,3,2)
        B,T,C,N = xyz.shape
        x = self.encode(xyz.reshape(B*T,C,N))
        x = x.reshape(B,T,-1)
        h0 = Variable(torch.zeros(2, x.size(0), 512)).to(x.device)
        out,hid = self.gru(x,h0)
        if self.f_only:
            return out.view(B,T,-1)
        x = F.relu((self.fc1(out)))
        x = F.relu((self.dropout(self.fc2(x))))
        result = self.fc3(x)
        return result.view(B,T,-1),out

class SMPL_trans(nn.Module):
    def __init__(self,in_c):

        super().__init__()

        self.in_fc = nn.Linear(in_c, 512)
        self.in_fc2 = nn.Linear(512, 128)
        cur_dim = 128

        self.decoder = TransformerDecoderLayer(d_model=128, nhead=8, cross_only=True)

        self.fc1 = nn.Linear(cur_dim, 64)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = 64
        
        self.fc2 = nn.Linear(cur_dim,32)

        self.t_fc2 = nn.Linear(32*24,512)
        self.t_fc3 = nn.Linear(512,256)
        self.t_fc4 = nn.Linear(256,3)


    def forward(self,pc,pred_kp,fea):

        B,T,_ = pred_kp.shape
        pred_kp = pred_kp.reshape(B*T,24,-1)
        pc = pc.reshape(B*T,256,-1)
        x1 = torch.cat((pred_kp,fea.repeat((1,1,24)).reshape(B*T,24,-1)),2)
        x2 = torch.cat((pc,fea.repeat((1,1,256)).reshape(B*T,256,-1)),2)

        x1 = F.relu(self.in_fc(x1))
        x2 = F.relu(self.in_fc(x2))

        x1 = self.in_fc2(x1)
        x2 = self.in_fc2(x2)
        x1 = x1.transpose(1,2).contiguous()
        x2 = x2.transpose(1,2).contiguous()
        x = self.decoder(query = x1, key = x2, query_pos=None, key_pos=None)    

        x = x.transpose(1,2).contiguous()
        x = F.relu((self.fc1(x)))

        x = self.fc2(x)
        x = x.reshape(B,T,-1)
        
        x = F.relu((self.t_fc2(x)))
        x = F.relu((self.dropout(self.t_fc3(x))))
        x = self.t_fc4(x)
        return x
    
class CPO(nn.Module):
    def __init__(self,in_c,hidden,out_c,kp_num,shape=False):

        super().__init__()

        self.kp_num=kp_num

        self.in_fc = nn.Linear(in_c, 1024)
        cur_dim = 1024
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)

        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        self.fc1 = nn.Linear(cur_dim, 24*256)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = hidden
        
        self.in_fc2 = nn.Linear(cur_dim,512)
        cur_dim = 512

        self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        self.spatial_net = nn.TransformerEncoder(ttf_layer,2)

        self.fc2 = nn.Linear(cur_dim,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,out_c) 

        self.shape=shape
        if shape:
            self.s_fc2 = nn.Linear(cur_dim,256)
            self.s_fc3 = nn.Linear(256,128)
            self.s_fc4 = nn.Linear(128,10)

    def forward(self,pred_kp,g_fea,l_fea=None,mask=None,mask_t=None):
        B,T,_ = pred_kp.shape
        x = torch.cat((pred_kp.transpose(0,1).contiguous(),g_fea.transpose(0,1).contiguous()),2) # T,B,72+1024
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc_t(x)
        if mask_t==None:
            x = self.temporal_net(x)
        else:
            x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32))         
        x = F.relu((self.fc1(x))) #T,B,24*256
        gl_fea = x
        if l_fea==None:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),g_fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1)),3)
        else:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)# 256+3+512
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x)
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        fea = x
        if self.shape:
            y = fea.transpose(0,1).contiguous().reshape(B,T*24,-1) #B*T*24*6
            y = torch.max(y, 1, keepdim=True)[0].reshape(B,-1)
            y = F.relu((self.s_fc2(y)))
            y = F.relu((self.dropout(self.s_fc3(y))))
            y = self.s_fc4(y)
            return y,fea.transpose(0,1).contiguous(),gl_fea.transpose(0,1).contiguous()
        else:
            y = None
        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return x.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous(),gl_fea.transpose(0,1).contiguous()