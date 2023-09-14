import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

def load_data(data_root, label_root):
    point_set = np.load(data_root, allow_pickle=True)
    label = np.load(label_root, allow_pickle=True)
    return point_set, label

class ModelNetDataLoaderC(Dataset):
    def __init__(self, data_root, label_root, npoint=1024, use_normals=False, partition='test'):
        assert partition in ['train', 'test']
        self.data, self.label = load_data(data_root, label_root)
        self.npoints = npoint
        self.use_normals = use_normals
        self.partition = partition

    def __getitem__(self, item):
        """Returns: point cloud as [N, 3] and its label as a scalar."""
        pc = self.data[item][:, :3]
        # print("pcc:",pc.shape)
        label = self.label[item]
        # print("labell:",label[0].shape)
        if self.use_normals:
            # pc = normalize_points_np(pc)
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return pc, label[0]

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser('training')
    # parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    # parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    # parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    # parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    # parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    args = parser.parse_args()

    # data = ModelNetDataLoader('/home/user_tp/workspace/data/modelnet40_normal_resampled',split='train', uniform=False, normal_channel=True,)
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    # for point,label in DataLoader:
    #     print(point.shape)
    #     print(label.shape)

    data_root = '/home/user_tp/workspace/data/ModelNet40-C/data_background_1.npy'
    label_root = '/home/user_tp/workspace/data/ModelNet40-C/label.npy'
    data_c = ModelNetDataLoaderC(data_root=data_root, label_root=label_root, args=args)
    print(data_c[0][0].shape)
    DataLoader_c = torch.utils.data.DataLoader(data_c, batch_size=24, shuffle=True)
    print("len:",len(DataLoader_c))