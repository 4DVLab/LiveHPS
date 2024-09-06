import argparse
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import *
from tqdm import tqdm
import torch
import numpy as np
from models import LiveHPS
import sys
sys.path.append("./smpl")
from smpl import SMPL, SMPL_MODEL_DIR
from dataset.Livehps_dataset import Dataset
from scipy.spatial.transform import Rotation as R

def gen_smpl(smpl,rot,shape,device):
    num = int(rot.shape[0]/shape.shape[0])
    rot = matrix_to_axis_angle(rotation_6d_to_matrix(rot).view(-1, 3,3)).reshape(-1, 72)
    pose_b = rot[:,3:].float()
    g_r = rot[:,:3].float()
    shape = shape.reshape(-1,1,10).repeat([1,num,1]).reshape(-1,10).float()
    zeros = np.zeros((g_r.shape[0], 3))
    transl_blob = torch.from_numpy(zeros).float().to(device)
    mesh = smpl(betas=shape,body_pose=pose_b,global_orient = g_r,transl=transl_blob)
    v = mesh.vertices.reshape(-1,6890,3) - mesh.joints.reshape(-1,24,3)[:,0,:].reshape(-1,1,3)
    j = mesh.joints.reshape(-1,24,3) - mesh.joints.reshape(-1,24,3)[:,0,:].reshape(-1,1,3)
    return v,j

def local2global(pose):
    kin_chains = [
        [20, 18, 16, 13, 9, 6, 3, 0],   # left arm
        [21, 19, 17, 14, 9, 6, 3, 0],   # right arm
        [7, 4, 1, 0],                   # left leg
        [8, 5, 2, 0],                   # right leg
        [12, 9, 6, 3, 0],               # head
        [0],                            # root, hip
    ]
    T = pose.shape[0]
    Rb2l = []
    cache = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    for chain in kin_chains:
        leaf_rotmat = torch.eye(3).unsqueeze(0).repeat(T,1,1)
        for joint in chain:
            joint_rotvec = pose[:, joint*3:joint*3+3]
            joint_rotmat = torch.from_numpy(R.from_rotvec(joint_rotvec.cpu()).as_matrix().astype(np.float32)).to("cpu")
            leaf_rotmat = torch.einsum("bmn,bnl->bml", joint_rotmat, leaf_rotmat)
            cache[joint] = leaf_rotmat
        Rb2l.append(leaf_rotmat)
    return cache

def cal_ang(gt_pose,pose):

    globalR = torch.from_numpy(pose[:, :3]).float()
    gt_matrix = local2global(gt_pose.reshape(-1,72))
    pose_matrix = local2global(torch.from_numpy(pose).reshape(-1,72))
    #print(gt_matrix)
    gt_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in gt_matrix if item!=None])).reshape(-1,3,3)
    pose_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in pose_matrix if item!=None])).reshape(-1,3,3)
    gt_axis = quaternion_to_axis_angle(matrix_to_quaternion(gt_matrix))
    pose_axis = quaternion_to_axis_angle(matrix_to_quaternion(pose_matrix))
    #print(gt_axis.shape)
    gt_norm = np.rad2deg(np.linalg.norm(gt_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    pose_norm = np.rad2deg(np.linalg.norm(pose_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    anger = np.abs((gt_norm-pose_norm)).mean(axis=1).mean()
    
    return anger

def jitter(pred):
    delta_t = 0.1
    pred_velo = (pred[:,1:,:,:]-pred[:,:-1,:,:]) / delta_t
    pred_acc = (pred_velo[:,1:,:,:]-pred_velo[:,:-1,:,:]) / delta_t
    jitter = torch.norm((pred_acc[:,1:,:,:]-pred_acc[:,:-1,:,:]) / delta_t, dim=-1)
    return torch.mean(jitter/100)

def test_one_epoch(model,device,test_loader,smpl):
    model.eval()
    test_loss = {
        'mpjpe':[],
        'mpvpe':[],
        'mpjpe-s':[],
        'mpvpe-s':[],
        'mpjpe-ts':[],
        'mpvpe-ts':[],
        'ang':[],
        'CD':[],
        'jitter':[]
    }
    for i,data in enumerate(tqdm(test_loader)):
        with torch.no_grad():  
            for j in range(3):
                seq_pc1 = data['data'+str(j+1)].to(device).float()
                B,T = seq_pc1.shape[0],seq_pc1.shape[1]
                seq_pose = data['gt_smpl'+str(j+1)].float().reshape(-1,T,3)
                gt_pose = torch.from_numpy(R.from_rotvec(seq_pose.reshape(-1,3)).as_matrix()).view(B,T,-1,3,3)
                gt_pose = matrix_to_rotation_6d(gt_pose).reshape(B*T,-1,6).to(device)
                gt_shape = data['shape'].to(device)
                seq_trans = data['T'+str(j+1)].to(device).float().reshape(-1,T,3)

                _,rot,shape,pre_trans = model(seq_pc1.float())

                final_pc = seq_pc1.reshape(B,T,256,3)-pre_trans.reshape(B,T,1,3)
                gt_v,gt_j = gen_smpl(smpl,gt_pose,gt_shape,device)
                pre_v,pre_j = gen_smpl(smpl,rot.reshape(B*T,-1,6),shape,device)
                pre_v_noshape,pre_j_noshape = gen_smpl(smpl,rot.reshape(B*T,-1,6),gt_shape,device)
                dif_tran = seq_trans.reshape(B,T,1,3)-pre_trans.reshape(B,T,1,3)

                loss1 = np.linalg.norm(pre_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                loss2 = np.linalg.norm(pre_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                loss11 = np.linalg.norm(pre_v_noshape[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2)#.mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                loss22 = np.linalg.norm(pre_j_noshape[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
                
                pose = R.from_matrix(rotation_6d_to_matrix(rot.view(B,T,24,6)).view(-1, 3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, 72)
                loss111 = np.linalg.norm((pre_v.reshape(B,T,-1,3)+dif_tran).reshape(B,T,-1)[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_v[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1)#.mean()#(loss_fn(out.float(),seq_gt.float()))
                loss222 = np.linalg.norm((pre_j.reshape(B,T,-1,3)+dif_tran).reshape(B,T,-1)[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3)-gt_j[:,:,:].cpu().detach().numpy().reshape(B*T,-1,3),axis=2).mean(axis=1)#.mean()#(loss_fn(out.float(),seq_gt.float()))
                
                an = cal_ang(seq_pose.reshape(-1,72),pose)
                loss3,_ = chamfer_distance(final_pc.reshape(B*T,256,3),pre_v.reshape(B*T,-1,3))
                loss_jitter = jitter(pre_j.reshape(B,T,24,3))

                test_loss['mpjpe-s'].append(loss2.item())
                test_loss['mpvpe-s'].append(loss1.item())
                test_loss['mpjpe'].append(loss22.item())
                test_loss['mpvpe'].append(loss11.mean(axis=1).mean().item())
                test_loss['mpjpe-ts'].append(loss222.mean().item())
                test_loss['mpvpe-ts'].append(loss111.mean().item())
                test_loss['ang'].append(an.mean().item())
                test_loss['CD'].append(loss3.mean().item())
                test_loss['jitter'].append(loss_jitter.item())
    loss_list = []
    for k in test_loss.keys():
        if len(test_loss[k])!=0:
            loss_list.append(np.array(test_loss[k]).mean())
            print(k,np.array(test_loss[k]).mean())
    print(f'{loss_list[0]*1000:.2f}/{loss_list[1]*1000:.2f}&{loss_list[2]*1000:.2f}/{loss_list[3]*1000:.2f}&{loss_list[4]*1000:.2f}/{loss_list[5]*1000:.2f}&{loss_list[6]:.2f}&{loss_list[7]*1000:.2f}/{loss_list[8]:.2f}')

    return np.array(test_loss['mpvpe']).mean(),np.array(test_loss['mpvpe-s']).mean(),np.array(test_loss['mpvpe-ts']).mean()

def options():
    parser = argparse.ArgumentParser(description='Baseline network')
    parser.add_argument('--save_path',type=str,default='')
    parser.add_argument('-n','--exp_name',type=str,default='')
    parser.add_argument('--root_dataset_path',type=str,default='')
    parser.add_argument('--train',type=bool,default=True)
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--num_points',type=int,default=256)

    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=200)

    parser.add_argument('--pretrained',default='')
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    return args

def main():
    args = options()
    device = args.device
    smpl = SMPL(SMPL_MODEL_DIR, create_transl=False).to(device)

    test_dataset = Dataset(args,'e')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,drop_last=False,pin_memory=False)
    model = LiveHPS()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(torch.cuda.device_count())
    model.to(device)

    test_one_epoch(model,device,test_loader,smpl)

if __name__ == "__main__":
    main()
