import torch
from .lib import PositionalEncoding,MLP
from torch import nn
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from torch.autograd import Variable
from .transformer_d import TransformerDecoderLayer

class model_tf(nn.Module):
    def __init__(self,k=36, channel=3):
        super(model_tf, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.gru = nn.GRU(1024,512,bidirectional=True,batch_first=True)
        #self.fc4 = nn.Linear(256,1)
        #self.sg = nn.Sigmoid()
        self.TF = TFEncoder()
        self.transformer_decoder = TransformerDecoderLayer(d_model=128, nhead=8, cross_only=True)
        #self.dropout = nn.Dropout(p=0.4)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        #self.relu = nn.ReLU()

    def encode(self, x):
        #x = x.permute(0,2,1)
        x, trans, trans_feat = self.feat(x)
        #x = F.log_softmax(x, dim=1)
        return x
    
    def forward(self,x,mask,device):
        xyz = x.permute(0,1,3,2)
        B,T,_,_ = xyz.shape
        x = torch.zeros((B,T,1024)).cuda()
        for i in range(T):
            x[:,i,:] = self.encode(xyz[:,i,:,:])
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2, x.size(0), 512)).to(device)
        #out,hid = self.gru(x,h0)
        out = self.TF(out,mask)
        #x = F.relu((self.fc1(out)))
        #x = F.relu((self.dropout(self.fc2(x))))
        #confidence = self.sg(self.fc4(x))
        #x = self.fc3(x)

        return out#.view(B,T,-1)#,confidence

class TFEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,out_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 1024)
        cur_dim = 1024
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, cur_dim*2, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.temporal_dim = cur_dim = 256

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
        self.fc1 = nn.Linear(cur_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 24*out_c)
        self.dropout = nn.Dropout(p=0.4)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self, pred_kp,fea):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        B,T,_ = pred_kp.shape
        x = torch.cat((pred_kp.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()),2) # T,B,72+(24*(3+9))+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc(x)
        x = self.temporal_net(x) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x.transpose(0,1).contiguous()
        data['context'] = x

class STFEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,out_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc2 = nn.Linear(in_c, 512)
        cur_dim = 512
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)

        self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        self.spatial_net = nn.TransformerEncoder(ttf_layer,2)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(cur_dim,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,out_c) 
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,fea,mask=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape

        if pred_kp.shape[2]==78:
            kp = pred_kp[:,:,6:72+6]
        elif pred_kp.shape[2]==78+30:
            kp = pred_kp[:,:,6:72+6]
        elif pred_kp.shape[2]==78+45:
            kp = pred_kp[:,:,6:72+6]
        elif pred_kp.shape[2]==72+45+72:
            kp = pred_kp[:,:,117:]
        elif pred_kp.shape[2]==72+45:
            kp = pred_kp[:,:,:72]
        elif pred_kp.shape[2]==72+30+72:
            kp = pred_kp[:,:,102:]
        elif pred_kp.shape[2]==72+30:
            kp = pred_kp[:,:,:72]
        else:
            kp = pred_kp
        x = torch.cat((kp.reshape(B,T,24,-1),fea.repeat((1,1,24)).reshape(B,T,24,1024)),3)
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 24,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return x.transpose(0,1).contiguous()
    
class TVTFEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,out_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        self.in_fc2 = nn.Linear(in_c, 512)
        cur_dim = 512

        max_freq = 10
        freq_scale = 0.1
        concat = True

        self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        self.spatial_net = nn.TransformerEncoder(ttf_layer,2)
        self.dropout = nn.Dropout(p=0.4)
        self.in_fc = nn.Linear(3*512,1024)
        cur_dim=1024
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        self.fc2 = nn.Linear(cur_dim,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,out_c) 

    def forward(self,kp_fea,mask=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T = kp_fea.shape[0],kp_fea.shape[1] #B,T,3,72+1024
        
        #x = torch.cat((pred_kp.reshape(B*T,3,72),fea.reshape((B*T,3,1024))),2).transpose(0,1).contiguous()# 3,B*T,1096
        x = kp_fea.reshape(B*T,-1,1024+78).transpose(0,1).contiguous()
        #x = torch.cat((pred_kp.reshape(B,T,24,-1),fea.repeat((1,1,24)).reshape(B,T,24,1024)),3)
        # B,T,24,32+3+1024
        #x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 3,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x)
        #x = self.spatial_net(x)
        x = x.transpose(0,1).contiguous().reshape(B,T,-1).transpose(0,1).contiguous()
        x = self.in_fc(x)
        x = self.pos_enc_t(x)
        x = self.temporal_net(x) # T,B,512

        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)

        return x.transpose(0,1).contiguous()
    
class MVEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,out_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        self.in_fc2 = nn.Linear(in_c, 512)
        cur_dim = 512

        max_freq = 10
        freq_scale = 0.1
        concat = True

        self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        self.spatial_net = nn.TransformerEncoder(ttf_layer,2)
        self.dropout = nn.Dropout(p=0.4)
        self.in_fc = nn.Linear(3*512,1024)
        cur_dim=1024
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        self.fc2 = nn.Linear(cur_dim,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,out_c) 

    def forward(self,pred_kp,fea,mask=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T = pred_kp.shape[0],pred_kp.shape[1] #B,T,72,3
        
        x = torch.cat((pred_kp.reshape(B*T,3,72),fea.reshape((B*T,3,1024))),2).transpose(0,1).contiguous()# 3,B*T,1096

        #x = torch.cat((pred_kp.reshape(B,T,24,-1),fea.repeat((1,1,24)).reshape(B,T,24,1024)),3)
        # B,T,24,32+3+1024
        #x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 24,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x)
        #x = self.spatial_net(x)
        x = x.transpose(0,1).contiguous().reshape(B,T,-1).transpose(0,1).contiguous()
        x = self.in_fc(x)
        x = self.pos_enc_t(x)
        x = self.temporal_net(x) # T,B,512

        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)

        return x.transpose(0,1).contiguous()
class TSTF_mk2m(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,kp_num,out_c=3,shape=False,joint=False):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 1024)
        cur_dim = 1024
        self.kp_num = kp_num
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
        self.fc1 = nn.Linear(cur_dim, kp_num*256)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = hidden
        
        self.in_fc2 = nn.Linear(cur_dim,512)
        cur_dim = 512

        self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        self.spatial_net = nn.TransformerEncoder(ttf_layer,2)

        # self.fc2 = nn.Linear(cur_dim,256)
        # self.fc3 = nn.Linear(256,128)
        # self.fc4 = nn.Linear(128,out_c)
        

        self.shape=shape
        if shape:
            self.s_fc2 = nn.Linear(32*24*6,1024)
            self.s_fc3 = nn.Linear(1024,256)
            self.s_fc4 = nn.Linear(256,10)
        self.joint=joint
        # if self.joint:
        #     self.fc2 = nn.Linear(cur_dim,256)
        #     self.fc3 = nn.Linear(256,128)
        #     self.fc4 = nn.Linear(128,out_c)
        # else:
        self.adaptive_A = nn.Sequential(
            nn.Linear(cur_dim, 6890, bias = True),
            )

        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,fea,l_fea=None,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        x = torch.cat((pred_kp.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()),2) # T,B,72+(24*(3+9))+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc_t(x)
        if mask_t==None:
            x = self.temporal_net(x)
        else:
            x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32)) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        x = F.relu((self.fc1(x))) #T,B,24*32
        if l_fea==None:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1)),3)
        else:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)# 256+3+256
        # B,T,64,32+3+1024
        x = x.reshape(B*T,self.kp_num,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 64,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
        if self.joint:
            x = x.transpose(0,1).contiguous().view(B,T,24,-1)
            x = torch.max(x, 1, keepdim=True)[0].reshape(B,24,-1)
            fea = x
            out = self.adaptive_A(x) #B,24,512
            return out.reshape(B,24,-1)
        else:
            fea = x
            # x = x.transpose(0,1).contiguous().view(B,T,24,-1)
            # x = torch.max(x, 1, keepdim=True)[0].reshape(B,24,-1)
            out = self.adaptive_A(fea)
        return out.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()
    
class TSTF_local2m(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,kp_num,out_c=3,ll=False,gl=False,shape=False,joint=False):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        # self.in_fc = nn.Linear(in_c, 1024)
        # cur_dim = 1024
        self.kp_num = kp_num
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        # self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # # transformer
        # tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        # self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
        # self.fc1 = nn.Linear(cur_dim, kp_num*256)
        # self.dropout = nn.Dropout(p=0.4)
        cur_dim = hidden
        
        self.in_fc2 = nn.Linear(cur_dim,512)
        cur_dim = 512
        self.ll = ll
        if ll:
            self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
            ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
            self.spatial_net = nn.TransformerEncoder(ttf_layer,2)

        # self.fc2 = nn.Linear(cur_dim,256)
        # self.fc3 = nn.Linear(256,128)
        # self.fc4 = nn.Linear(128,out_c)
        

        self.shape=shape
        if shape:
            self.s_fc2 = nn.Linear(32*24*6,1024)
            self.s_fc3 = nn.Linear(1024,256)
            self.s_fc4 = nn.Linear(256,10)
        self.joint=joint
        # if self.joint:
        #     self.fc2 = nn.Linear(cur_dim,256)
        #     self.fc3 = nn.Linear(256,128)
        #     self.fc4 = nn.Linear(128,out_c)
        # else:
        self.adaptive_A = nn.Sequential(
            nn.Linear(cur_dim, 6890, bias = True),
            )

        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,fea,l_fea=None,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        x = torch.cat((pred_kp.reshape(B,T,self.kp_num,-1),fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        x = self.in_fc2(x)
        if self.ll:
            x = self.pos_enc_s(x) # 64,B*T,512
            x = self.spatial_net(x)
        if self.joint:
            x = x.transpose(0,1).contiguous().view(B,T,24,-1)
            x = torch.max(x, 1, keepdim=True)[0].reshape(B,24,-1)
            fea = x
            out = self.adaptive_A(x) #B,24,512
            return out.reshape(B,24,-1)
        else:
            fea = x
            # x = x.transpose(0,1).contiguous().view(B,T,24,-1)
            # x = torch.max(x, 1, keepdim=True)[0].reshape(B,24,-1)
            out = self.adaptive_A(fea)
        return out.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()
    

class TSTF_pc2m(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,kp_num,shape):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 1024)
        cur_dim = 1024
        self.kp_num = kp_num
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
        # self.fc1 = nn.Linear(cur_dim, kp_num*256)
        # self.dropout = nn.Dropout(p=0.4)
        # cur_dim = hidden
        
        self.in_fc2 = nn.Linear(cur_dim,512)
        cur_dim = 512

        # self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        # self.spatial_net = nn.TransformerEncoder(ttf_layer,2)

        # self.fc2 = nn.Linear(cur_dim,256)
        # self.fc3 = nn.Linear(256,128)
        # self.fc4 = nn.Linear(128,out_c) 
        self.adaptive_A = nn.Sequential(
            nn.Linear(cur_dim, 6890, bias = True),
            )

        self.shape=shape
        if shape:
            self.s_fc2 = nn.Linear(32*24*6,1024)
            self.s_fc3 = nn.Linear(1024,256)
            self.s_fc4 = nn.Linear(256,10)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,fea,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T = fea.shape[0],fea.shape[1]
        x = fea.reshape(B,T,-1).transpose(0,1).contiguous()# T,B,1024
        # x = torch.cat((pred_kp.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()),2) # T,B,72+(24*(3+9))+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc_t(x)
        if mask_t==None:
            x = self.temporal_net(x)
        else:
            x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32)) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        # x = F.relu((self.fc1(x))) #T,B,24*32

        # x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1)),3)
        # # B,T,64,32+3+1024
        # x = x.reshape(B*T,self.kp_num,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        # x = self.pos_enc_s(x) # 64,B*T,512
        # if mask == None:
        #     x = self.spatial_net(x)
        # else:
        #     x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
        fea = x
        ad = self.adaptive_A(fea)
        
        # x = F.relu((self.fc2(x)))
        # x = F.relu((self.dropout(self.fc3(x))))
        # x = self.fc4(x)
        return ad.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()

class TSTF_select2(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.in_fc = nn.Linear(3, 64)
        cur_dim = 64
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        # self.conv1 = nn.Conv1d(in_channels=63, out_channels=128, kernel_size=1)
        # Add more conv layers if needed
        self.fc1 = nn.Linear(64, 32)  # Fully connected layer
        self.fc2 = nn.Linear(32, 1) 


        # self.t_fc2 = nn.Linear(32*24,512)
        # self.t_fc3 = nn.Linear(512,256)
        # self.t_fc4 = nn.Linear(256,3)


    def forward(self,pred_kp):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        pred_kp = pred_kp.reshape(B,T,24,63,3).transpose(0,1).contiguous().reshape(T,B*24*63,3)
        x = self.in_fc(pred_kp)
        x = self.pos_enc_t(x)
        x = self.temporal_net(x)

        x = x.reshape(T,B,24,63,64).transpose(0,1)#.reshape(-1,512) #B,T,24,512
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.reshape(B,T,24,63)
        # return F.softmax(x, dim=-1) 


class TSTF_select(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,kp=True):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 512)
        self.in_fc2 = nn.Linear(1024+3, 512)
        # self.in_fc2 = nn.Linear(512, 128)
        cur_dim = 512

        self.decoder = TransformerDecoderLayer(d_model=512, nhead=8, cross_only=True)

        if kp:
            max_freq = 10
            freq_scale = 0.1
            concat = True
            self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
            # transformer
            tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
            self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        self.fc1 = nn.Linear(cur_dim, 128)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = 128
        self.fc2 = nn.Linear(cur_dim,3)

    def forward(self,pc,pred_kp,fea,tran=False,num=24):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,C = pred_kp.shape
        if C==63*3 or C==3:
            num = 1
        else:
            num = 24
        if tran:
            num=1
        pred_kp = pred_kp.reshape(B*T,num,-1)
        pc = pc.reshape(B*T,256,-1)
        x1 = torch.cat((pred_kp,fea.repeat((1,1,num)).reshape(B*T,num,-1)),2) # BT,24,3+1024
        x2 = torch.cat((pc,fea.repeat((1,1,256)).reshape(B*T,256,-1)),2) # BT,256,3+1024
        # x1 = pred_kp
        # x2 = pc.reshape(B)
        x1 = F.relu(self.in_fc(x1)) # BT,24,512
        x2 = F.relu(self.in_fc2(x2))

        # x1 = self.in_fc2(x1)# BT,24,128
        # x2 = self.in_fc2(x2)
        x1 = x1.transpose(1,2).contiguous()
        x2 = x2.transpose(1,2).contiguous()
        x = self.decoder(query = x1, key = x2, query_pos=None, key_pos=None)    
        x = x.transpose(1,2).contiguous().reshape(B*T,num,512)# BT,24,128  

        if not tran:
            x = x.transpose(0,1).contiguous()
            x = self.pos_enc_t(x) # 24,B*T,512
            x = self.temporal_net(x)
            x = x.transpose(0,1).contiguous()

        loc_fea = x.reshape(B,T,num,512)
        
        x = F.relu((self.fc1(x))) #BT,24,64
        x = self.fc2(x)#BT,24,32
        x = x.reshape(B,T,-1) #B,T,24*3
        return x,loc_fea

class TSTF_pos_10230(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 512)
        # self.in_fc2 = nn.Linear(512, 128)
        cur_dim = 512

        self.decoder = TransformerDecoderLayer(d_model=512, nhead=8, cross_only=True)

        # self.pos_fc = nn.Linear(640, 512)
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        self.fc1 = nn.Linear(cur_dim, 128)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = 128
        self.fc2 = nn.Linear(cur_dim,2)

        self.in_fc3 = nn.Linear(3,64)
        cur_dim = 64
        self.pos_enc_t2 = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers2 = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net2 = nn.TransformerEncoder(tf_layers2, 8)
        self.fc3 = nn.Linear(64,3)

        # cur_dim = 24*3*3
        # self.pos_enc_t2 = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # # transformer
        # tf_layers2 = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        # self.temporal_net2 = nn.TransformerEncoder(tf_layers2, 8)

        # self.select_fc = nn.Linear(24*3*3,24*3)
        
        # self.fc_cf1_1 = nn.Linear(512,128)
        # self.fc_cf1_2 = nn.Linear(128,1)
        # self.fc_cf2_1 = nn.Linear(512,128)
        # self.fc_cf2_2 = nn.Linear(128,1)


        # self.t_fc2 = nn.Linear(32*24,512)
        # self.t_fc3 = nn.Linear(512,256)
        # self.t_fc4 = nn.Linear(256,3)


    def forward(self,pc,pred_kp,fea):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        pred_kp = pred_kp.reshape(B*T,24,-1)
        pc = pc.reshape(B*T,256,-1)
        x1 = torch.cat((pred_kp,fea.repeat((1,1,24)).reshape(B*T,24,-1)),2) # BT,24,3+1024
        x2 = torch.cat((pc,fea.repeat((1,1,256)).reshape(B*T,256,-1)),2) # BT,256,3+1024

        x1 = F.relu(self.in_fc(x1)) # BT,24,512
        x2 = F.relu(self.in_fc(x2))

        # x1 = self.in_fc2(x1)# BT,24,128
        # x2 = self.in_fc2(x2)
        x1 = x1.transpose(1,2).contiguous()
        x2 = x2.transpose(1,2).contiguous()
        x = self.decoder(query = x1, key = x2, query_pos=None, key_pos=None)    
        x = x.transpose(1,2).contiguous().reshape(B,T,24,512)# BT,24,128
        # x = torch.cat((x,pos_ec.repeat((1,1,24)).reshape(B,T,24,128)),dim=-1) #B,T,24,640
        # cf_kp = F.relu(self.fc_cf1_1(x))
        # cf_kp = self.fc_cf1_2(cf_kp)
        
        x = x[:,1:] - x[:,:-1]
        # x = self.pos_fc(x)

        x = x.transpose(1,2).contiguous().reshape(B*24,-1,512).transpose(0,1).contiguous()
        x = self.pos_enc_t(x)
        x = self.temporal_net(x) # T,B,1024  
        x = x.transpose(0,1).contiguous().reshape(B,24,-1,512).transpose(1,2).contiguous().reshape(B*(T-1),-1,512)
        # x = x.transpose(0,1).contiguous().reshape(T,B*24,512)#.transpose(0,1).contiguous()
        # x = self.pos_enc_t(x)
        # x = self.temporal_net(x) # T,B,1024  
        # x = x.reshape(T,B,24,512).transpose(0,1).contiguous().reshape(B*T,24,512)
        # cf_con = F.relu(self.fc_cf2_1(x))
        # cf_con = self.fc_cf2_2(cf_con)
        x = F.relu((self.fc1(x))) #BT,24,64
        x = self.fc2(x)#BT,24,32
        x = x.reshape(B,T-1,24,2) #B,T,24*3

        new_trj = pred_kp.reshape(B,T,24,3)[:,1:] - pred_kp.reshape(B,T,24,3)[:,:-1]
        
        # print(x[:,:,:,0].shape,new_trj.shape)

        pred_kp2 = pred_kp.reshape(B,T,24,3)[:,:-1]+x[:,:,:,0].reshape(B,T-1,24,1)*new_trj
        pred_kp3 = pred_kp2+x[:,:,:,1].reshape(B,T-1,24,1)*new_trj
        new_kp = torch.cat((pred_kp.reshape(B,T,24,3)[:,:-1],pred_kp2,pred_kp3),dim=3).reshape(B,T-1,24,3,3).transpose(2,3).reshape(B,(T-1)*3,24,3)
        new_kp = torch.cat((new_kp,pred_kp.reshape(B,T,24,3)[:,-1].reshape(B,1,24,3)),dim=1)

        # new_kp = new_kp.transpose(0,1).contiguous().reshape(-1,B*24,3)
        # new_kp = self.in_fc3(new_kp)
        # new_kp = self.pos_enc_t2(new_kp)
        # new_kp = self.temporal_net2(new_kp)
        # new_kp = new_kp.reshape(-1,B,24,64).transpose(0,1)
        # new_kp = self.fc3(new_kp)
        
        # new_kp = self.fc3(new_kp.reshape(-1,24,3))

        return new_kp#,pred_kp#,cf_kp,cf_con

class TSTF_pos_ec(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,out_c=3,pos_ec=True):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 512)
        # self.in_fc2 = nn.Linear(512, 128)
        cur_dim = 512

        self.decoder = TransformerDecoderLayer(d_model=512, nhead=8, cross_only=True)

        if pos_ec:
            self.pos_fc = nn.Linear(640, 512)
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        self.fc1 = nn.Linear(cur_dim, 128)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = 128
        self.fc2 = nn.Linear(cur_dim,out_c)

        # cur_dim = 24*3*3
        # self.pos_enc_t2 = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # # transformer
        # tf_layers2 = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        # self.temporal_net2 = nn.TransformerEncoder(tf_layers2, 8)

        # self.select_fc = nn.Linear(24*3*3,24*3)
        
        # self.fc_cf1_1 = nn.Linear(512,128)
        # self.fc_cf1_2 = nn.Linear(128,1)
        # self.fc_cf2_1 = nn.Linear(512,128)
        # self.fc_cf2_2 = nn.Linear(128,1)


        # self.t_fc2 = nn.Linear(32*24,512)
        # self.t_fc3 = nn.Linear(512,256)
        # self.t_fc4 = nn.Linear(256,3)


    def forward(self,pc,pred_kp,fea,pos_ec=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,C = pred_kp.shape
        num = int(C/3)
        pred_kp = pred_kp.reshape(B*T,num,-1)
        pc = pc.reshape(B*T,256,-1)
        x1 = torch.cat((pred_kp,fea.repeat((1,1,num)).reshape(B*T,num,-1)),2) # BT,24,3+1024
        x2 = torch.cat((pc,fea.repeat((1,1,256)).reshape(B*T,256,-1)),2) # BT,256,3+1024

        x1 = F.relu(self.in_fc(x1)) #BT,24,512
        x2 = F.relu(self.in_fc(x2))

        # x1 = self.in_fc2(x1) #BT,24,128
        # x2 = self.in_fc2(x2)
        x1 = x1.transpose(1,2).contiguous()
        x2 = x2.transpose(1,2).contiguous()
        x = self.decoder(query = x1, key = x2, query_pos=None, key_pos=None)    
        x = x.transpose(1,2).contiguous().reshape(B,T,num,512)# BT,24,128
        if pos_ec!=None:
            x = torch.cat((x,pos_ec.repeat((1,1,num)).reshape(B,T,num,128)),dim=-1) #B,T,24,640
        # cf_kp = F.relu(self.fc_cf1_1(x))
        # cf_kp = self.fc_cf1_2(cf_kp)
        
        x = x[:,1:] - x[:,:-1]
        if pos_ec!=None:
            x = self.pos_fc(x)

        x = x.transpose(1,2).contiguous().reshape(B*num,-1,512).transpose(0,1).contiguous()
        x = self.pos_enc_t(x)
        x = self.temporal_net(x) # T,B,1024  
        x = x.transpose(0,1).contiguous().reshape(B,num,-1,512).transpose(1,2).contiguous().reshape(B*(T-1),-1,512)
        # x = x.transpose(0,1).contiguous().reshape(T,B*24,512)#.transpose(0,1).contiguous()
        # x = self.pos_enc_t(x)
        # x = self.temporal_net(x) # T,B,1024  
        # x = x.reshape(T,B,24,512).transpose(0,1).contiguous().reshape(B*T,24,512)
        # cf_con = F.relu(self.fc_cf2_1(x))
        # cf_con = self.fc_cf2_2(cf_con)
        x = F.relu((self.fc1(x)))#BT,24,64
        x = self.fc2(x)#BT,24,32
        x = x.reshape(B,T-1,-1)#B,T,24*3
        # for step in range(2):
        # for step in range(4):
        #     kp_p = pred_kp.clone()
        #     kp_m = pred_kp.clone()
        #     kp_p.reshape(B,T,24,3)[:,1:] = kp_p.reshape(B,T,24,3)[:,:-1]+x.reshape(B,T-1,24,3)
        #     kp_m.reshape(B,T,24,3)[:,:-1] = kp_m.reshape(B,T,24,3)[:,1:]-x.reshape(B,T-1,24,3)
        #     pred_kp = torch.cat((pred_kp.reshape(B,T,24,-1,3),kp_p.reshape(B,T,24,1,3),kp_m.reshape(B,T,24,1,3)),3).reshape(B,T,-1)
        #     pred_kp = pred_kp.transpose(0,1).contiguous()
        #     pred_kp = self.pos_enc_t2(pred_kp)
        #     pred_kp = self.temporal_net2(pred_kp)
        #     pred_kp = pred_kp.transpose(0,1).contiguous()
        #     pred_kp = self.select_fc(pred_kp) 
        # x = F.relu((self.t_fc2(x)))
        # x = F.relu((self.dropout(self.t_fc3(x))))
        # x = self.t_fc4(x)#B,T,3
        return x#,pred_kp#,cf_kp,cf_con
   
class TSTF_sk2T(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
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
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        pred_kp = pred_kp.reshape(B*T,24,-1)
        pc = pc.reshape(B*T,256,-1)
        x1 = torch.cat((pred_kp,fea.repeat((1,1,24)).reshape(B*T,24,-1)),2) # BT,24,3+1024
        x2 = torch.cat((pc,fea.repeat((1,1,256)).reshape(B*T,256,-1)),2) # BT,256,3+1024

        x1 = F.relu(self.in_fc(x1)) # BT,24,512
        x2 = F.relu(self.in_fc(x2))

        x1 = self.in_fc2(x1)# BT,24,128
        x2 = self.in_fc2(x2)
        x1 = x1.transpose(1,2).contiguous()
        x2 = x2.transpose(1,2).contiguous()
        x = self.decoder(query = x1, key = x2, query_pos=None, key_pos=None) # BT,24,128      

        x = x.transpose(1,2).contiguous()
        x = F.relu((self.fc1(x))) #BT,24,64

        x = self.fc2(x)#BT,24,32
        x = x.reshape(B,T,-1) #B,T,24*32
        
        x = F.relu((self.t_fc2(x)))
        x = F.relu((self.dropout(self.t_fc3(x))))
        x = self.t_fc4(x)#B,T,3

        return x
    
class TSTFEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,out_c,shape):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc = nn.Linear(in_c, 1024)
        cur_dim = 1024
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
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
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,fea,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        x = torch.cat((pred_kp.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()),2) # T,B,72+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc_t(x)
        if mask_t==None:
            x = self.temporal_net(x)
        else:
            x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32)) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        x = F.relu((self.fc1(x))) #T,B,24*256

        if pred_kp.shape[2]==78:
            kp = pred_kp[:,:,6:72+6]
        elif pred_kp.shape[2]==78+30:
            kp = pred_kp[:,:,6:72+6]
        elif pred_kp.shape[2]==78+45:
            kp = pred_kp[:,:,6:72+6]
        elif pred_kp.shape[2]==72+45+72:
            kp = pred_kp[:,:,117:]
        elif pred_kp.shape[2]==72+45:
            kp = pred_kp[:,:,:72]
        elif pred_kp.shape[2]==72+30+72:
            kp = pred_kp[:,:,102:]
        elif pred_kp.shape[2]==72+30:
            kp = pred_kp[:,:,:72]
        else:
            kp = pred_kp
        # print(x.shape,kp.shape,fea.shape)
        x = torch.cat((x.reshape(T,B,24,-1).transpose(0,1).contiguous(),kp.reshape(B,T,24,-1),fea.repeat((1,1,24)).reshape(B,T,24,-1)),3)
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 24,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
        fea = x
        if self.shape:
            y = fea.transpose(0,1).contiguous().reshape(B,T*24,-1) #B*T*24*6
            y = torch.max(y, 1, keepdim=True)[0].reshape(B,-1)
            y = F.relu((self.s_fc2(y)))
            y = F.relu((self.dropout(self.s_fc3(y))))
            y = self.s_fc4(y)
            return y,fea.transpose(0,1).contiguous()
        else:
            y = None
        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return x.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()


class TSTF_vs2r(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,out_c,kp_num,shape=False):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        # self.dp_fc = nn.Linear(in_c, 1024)
        # self.dp_fc2 = nn.Linear(1024, 512)
        # self.dp_fc3 = nn.Linear(512, 128)
        self.kp_num=kp_num

        self.in_fc = nn.Linear(in_c, 1024)
        cur_dim = 1024
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
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
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,g_fea,l_fea=None,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        # vs = pred_vs.transpose(2,3).reshape(B*T,3,-1)
        # vs = F.relu((self.dp_fc(vs)))
        # vs = F.relu((self.dropout(self.dp_fc2(vs))))
        # vs = self.dp_fc3(vs)
        # vs = vs.transpose(1,2).reshape(B,T,-1)#B,T,128,3
        x = torch.cat((pred_kp.transpose(0,1).contiguous(),g_fea.transpose(0,1).contiguous()),2) # T,B,72+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc_t(x)
        if mask_t==None:
            x = self.temporal_net(x)
        else:
            x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32)) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        x = F.relu((self.fc1(x))) #T,B,24*256
        gl_fea = x

        # print(x.shape,kp.shape,fea.shape)
        if l_fea==None:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),g_fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1)),3)
        else:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)# 256+3+512
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 24,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
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
    
class TSTF_vs2r_v2(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,out_c,kp_num,shape=False):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.dp_fc = nn.Linear(in_c, 1024)
        self.dp_fc2 = nn.Linear(1024, 512)
        self.dp_fc3 = nn.Linear(512, 128)
        self.kp_num=kp_num

        self.in_fc = nn.Linear(1024+72+384, 1024)
        cur_dim = 1024
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
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
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_vs,pred_kp,g_fea,l_fea=None,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        vs = pred_vs.transpose(2,3).reshape(B*T,3,-1)
        vs = F.relu((self.dp_fc(vs)))
        vs = F.relu((self.dropout(self.dp_fc2(vs))))
        vs = self.dp_fc3(vs)
        vs = vs.transpose(1,2).reshape(B,T,-1)#B,T,128,3
        x = torch.cat((vs.transpose(0,1).contiguous(),pred_kp.transpose(0,1).contiguous(),g_fea.transpose(0,1).contiguous()),2) # T,B,72+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        x = self.in_fc(x) # T,B,1024

        x = self.pos_enc_t(x)
        if mask_t==None:
            x = self.temporal_net(x)
        else:
            x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32)) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        x = F.relu((self.fc1(x))) #T,B,24*256
        gl_fea = x

        # print(x.shape,kp.shape,fea.shape)
        if l_fea==None:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1)),3)
        else:
            x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)# 256+3+512
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 24,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
        fea = x
        if self.shape:
            y = fea.transpose(0,1).contiguous().reshape(B,T*24,-1) #B*T*24*6
            y = torch.max(y, 1, keepdim=True)[0].reshape(B,-1)
            y = F.relu((self.s_fc2(y)))
            y = F.relu((self.dropout(self.s_fc3(y))))
            y = self.s_fc4(y)
            return y,fea.transpose(0,1).contiguous()
        else:
            y = None
        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return x.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous(),gl_fea.transpose(0,1).contiguous()

class TSTF_local2r(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,hidden,out_c,kp_num,ll=False,shape=False):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        # self.dp_fc = nn.Linear(in_c, 1024)
        # self.dp_fc2 = nn.Linear(1024, 512)
        # self.dp_fc3 = nn.Linear(512, 128)
        self.kp_num=kp_num

        # self.in_fc = nn.Linear(2048+72, 2048)
        # cur_dim = 2048
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        # self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # # transformer
        # tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        # self.temporal_net = nn.TransformerEncoder(tf_layers, 2)
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)
        # self.fc1 = nn.Linear(cur_dim, 24*256)
        self.dropout = nn.Dropout(p=0.4)
        cur_dim = hidden
        
        self.in_fc2 = nn.Linear(cur_dim,512)
        cur_dim = 512

        self.ll = ll
        if ll:
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
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,g_fea,l_fea=None,mask=None,mask_t=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        # vs = pred_vs.transpose(2,3).reshape(B*T,3,-1)
        # vs = F.relu((self.dp_fc(vs)))
        # vs = F.relu((self.dropout(self.dp_fc2(vs))))
        # vs = self.dp_fc3(vs)
        # vs = vs.transpose(1,2).reshape(B,T,-1)#B,T,128,3
        # x = torch.cat((pred_kp.transpose(0,1).contiguous(),g_fea.transpose(0,1).contiguous()),2) # T,B,72+1024
        #mask = mask.transpose(0,1).contiguous()
        #if self.in_mlp is not None:
        #    x = self.in_mlp(x)
        #if self.in_fc is not None:
        # x = self.in_fc(x) # T,B,1024

        # x = self.pos_enc_t(x)
        # if mask_t==None:
        #     x = self.temporal_net(x)
        # else:
        #     x = self.temporal_net(x,src_key_padding_mask=mask_t.reshape(-1,32)) # T,B,1024           

        #x = torch.cat((x,fea.transpose(0,1).contiguous()),2)
        # x = F.relu((self.fc1(x))) #T,B,24*256
        # gl_fea = x

        x = torch.cat((pred_kp.reshape(B,T,self.kp_num,-1),g_fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)
        # print(x.shape,kp.shape,fea.shape)
        # if l_fea==None:
        #     x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),fea.repeat((1,1,self.kp_num)).reshape(B,T,self.kp_num,-1)),3)
        # else:
        #     x = torch.cat((x.reshape(T,B,self.kp_num,-1).transpose(0,1).contiguous(),pred_kp.reshape(B,T,self.kp_num,-1),l_fea.reshape(B,T,self.kp_num,-1)),3)# 256+3+512
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        
        x = self.in_fc2(x)
        if self.ll:
            x = self.pos_enc_s(x) # 24,B*T,512
        # if mask == None:
        #     x = self.spatial_net(x)
        # else:
        #     x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
            x = self.spatial_net(x)
        fea = x
        if self.shape:
            y = fea.transpose(0,1).contiguous().reshape(B,T*24,-1) #B*T*24*6
            y = torch.max(y, 1, keepdim=True)[0].reshape(B,-1)
            y = F.relu((self.s_fc2(y)))
            y = F.relu((self.dropout(self.s_fc3(y))))
            y = self.s_fc4(y)
            return y,fea.transpose(0,1).contiguous()
        else:
            y = None
        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return x.transpose(0,1).contiguous(),fea.transpose(0,1).contiguous()#,gl_fea.transpose(0,1).contiguous()

class STTFEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self,in_c,out_c):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()

        """ in MLP """
        #if 'in_mlp' in specs:
        #    in_mlp_cfg = specs['in_mlp']
        #    self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.in_mlp.out_dim
        #else:
        #    self.in_mlp = None
        
        #if 'in_fc' in specs:
        self.in_fc2 = nn.Linear(in_c, 512)
        cur_dim = 512
        #else:
        #    self.in_fc = None

        """ temporal network """
        #temporal_cfg = specs['transformer']
        # positional encoding
        #pe_cfg = specs['transformer']['positional_encoding']
        max_freq = 10
        freq_scale = 0.1
        concat = True
        #self.spatial_dim = cur_dim = 32

        """ out MLP """
        #if 'out_mlp' in specs:
        #    out_mlp_cfg = specs['out_mlp']
        #    self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
        #    cur_dim = self.out_mlp.out_dim
        #else:
        #self.out_mlp = None

        #if 'context_dim' in specs:
        #self.fc = nn.Linear(cur_dim, 24*3+6)

        self.pos_enc_s = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        ttf_layer = nn.TransformerEncoderLayer(cur_dim,8,2*cur_dim,0.1)
        self.spatial_net = nn.TransformerEncoder(ttf_layer,2)
        self.dropout = nn.Dropout(p=0.4)
        # self.in_fc = nn.Linear(24*512,1024)
        cur_dim=512
        # self.pos_enc_t = PositionalEncoding(cur_dim, cur_dim, 'original', max_freq, freq_scale, concat=concat)
        # # transformer
        # tf_layers = nn.TransformerEncoderLayer(cur_dim, 8, 2*cur_dim, 0.1)
        # self.temporal_net = nn.TransformerEncoder(tf_layers, 2)

        self.fc2 = nn.Linear(cur_dim,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,out_c) 
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #elf.fc4 = nn.Linear(cur_dim, 512)
        #self.fc5 = nn.Linear(512, 256)
        #self.fc6 = nn.Linear(256, 24*3)
        #cur_dim = specs['context_dim']
        #else:
        #    self.fc = None
        #ctx['context_dim'] = cur_dim

    def forward(self,pred_kp,fea,mask=None):
        #x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        #BT,jn,ch = pred_kp.shape(-1,24,3)
        B,T,_ = pred_kp.shape
        x = torch.cat((pred_kp.reshape(B,T,24,-1),fea.repeat((1,1,24)).reshape(B,T,24,1024)),3)
        # B,T,24,32+3+1024
        x = x.reshape(B*T,24,-1).transpose(0,1).contiguous()
        #x = x.transpose(0,1).contiguous().reshape(32,32,-1) # B,T,24*64
        #x = x.transpose(0,1).contiguous()
        x = self.in_fc2(x)
        x = self.pos_enc_s(x) # 24,B*T,512
        if mask == None:
            x = self.spatial_net(x)
        else:
            x = self.spatial_net(x,src_key_padding_mask=mask.reshape(-1,24))
        #x = self.spatial_net(x)
        x = x.transpose(0,1).contiguous().reshape(B,T,24,-1)#.transpose(0,1).contiguous()
        # x = F.relu(self.in_fc(x))
        # x = self.pos_enc_t(x)
        # x = self.temporal_net(x) # T,B,512
        x = F.relu((self.fc2(x)))
        x = F.relu((self.dropout(self.fc3(x))))
        x = self.fc4(x)

        return x#.transpose(0,1).contiguous()