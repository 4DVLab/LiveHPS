import torch
import pickle
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import copy

def farthest_point_sample(xyz, npoint):
    ndataset = xyz.shape[0]
    if ndataset<npoint:
        repeat_n = int(npoint/ndataset)
        xyz = np.tile(xyz,(repeat_n,1))
        xyz = np.append(xyz,xyz[:npoint%ndataset],axis=0)
        return xyz
    centroids = np.zeros(npoint)
    distance = np.ones(ndataset) * 1e10
    farthest =  np.random.randint(0, ndataset)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[int(farthest)]
        dist = np.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return xyz[np.int32(centroids)]

class Dataset(torch.utils.data.Dataset):
    def __init__(self,args,m):
        self.dataset = []
        self.type = type
        if m == 'e':
            data_info_path = './data/fm_test2.pkl'
        self.Train = m
        self.num_points = args.num_points
        T = args.frames
        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        
        file.close()
        old_motion_id = datas[0]['motion_id']
        seq = []
        j = 0
        if T == 1:
            self.dataset = datas
        else:
            while True:
                if j>len(datas)-1:
                    break
                motion_id = datas[j]['motion_id']
                if motion_id==old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq = []
                    seq.append(datas[j])
                    j+=1
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]
        print(len(self.dataset))

    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = dict()
        seq_pc['pc1'] = []
        seq_pc['pc2'] = []
        seq_pc['pc3'] = []
        seq_gt = []
        seq_t = dict()
        seq_t['pc1'] = []
        seq_t['pc2'] = []
        seq_t['pc3'] = []
        seq_loc = []
        seq_pose1 = []
        seq_pose2 = []
        seq_pose3 = []
        rot1 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        rot2 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        for i in range(len(example_seq)):
            example = example_seq[i]
            for i in range(3):
                if 'pc_'+str(i+1) not in example.keys() and i>1:
                    continue
                else:
                    if 'pc_'+str(i+1) not in example.keys():
                        pc_data = example['pc']
                    else:
                        pc_data = example['pc_'+str(i+1)]
                if 'T_1' in example.keys():
                    tran = example['T_1']
                else:
                    tran = np.array([0,0,0])
                # if i == 0:
                #     if len(pc_data)<=1:
                #         pc_data = tran.reshape(1,3)
                if i == 2:
                    if 'rot2' not in example.keys():
                        rot2 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
                    else:
                        rot2 = example['rot2']
                    if 'T_3' in example.keys():
                        tran = example['T_3']
                    else:
                        tran=np.array([0,0,0])
                    tran = np.matmul(rot2,tran.T).T
                    # if len(pc_data)<=1:
                    #     pc_data = tran.reshape(1,3)
                    # else:
                    #     pc_data = np.matmul(rot2,pc_data.T).T
                    pc_data = np.matmul(rot2,pc_data.T).T
                if i == 1:
                    if 'T_2' in example.keys():
                        tran = example['T_2']
                    else:
                        tran=np.array([0,0,0])
                    if 'rot1' in example.keys():
                        rot1 = example['rot1']
                    else:
                        rot1 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
                    tran = np.matmul(rot1,tran.T).T
                    # if len(pc_data)<=1:
                    #     pc_data = tran.reshape(1,3)
                    # else:
                    #     pc_data = np.matmul(rot1,pc_data.T).T
                    pc_data = np.matmul(rot1,pc_data.T).T

                loc = pc_data.mean(0)
                pc_data = pc_data - loc
                pc_data = farthest_point_sample(pc_data,256)

                seq_pc['pc'+str(i+1)].append(pc_data)
                seq_t['pc'+str(i+1)].append(tran-loc)

            pose = example['gt']
            seq_pose1.append(pose)
            pose2 = copy.deepcopy(pose)
            gt_r2 = R.from_matrix(np.matmul(rot1,R.from_rotvec(pose2[:3]).as_matrix())).as_rotvec()
            pose2[:3] = gt_r2
            pose3 = copy.deepcopy(pose)
            gt_r3 = R.from_matrix(np.matmul(rot2,R.from_rotvec(pose3[:3]).as_matrix())).as_rotvec()
            seq_pose2.append(pose2)
            pose3[:3] = gt_r3
            seq_pose3.append(pose3)

        if 'shape' in example.keys():
            shape = example['shape']
        else:
            shape = np.loadtxt('/sharedata/home/renym/mocap-shape/smpl/shape.txt')
        Item = {
            'data1':np.array(seq_pc['pc1']),
            'data2':np.array(seq_pc['pc2']),
            'data3':np.array(seq_pc['pc3']),
            'gt' : np.array(seq_gt),
            'T1': np.array(seq_t['pc1']),
            'T2': np.array(seq_t['pc2']),
            'T3': np.array(seq_t['pc3']),
            'loc': np.array(seq_loc),
            'gt_smpl1':np.array(seq_pose1),
            'gt_smpl2':np.array(seq_pose2),
            'gt_smpl3':np.array(seq_pose3),
            'shape':shape,
        }
        return Item
    
    def __len__(self):
        return len(self.dataset)
