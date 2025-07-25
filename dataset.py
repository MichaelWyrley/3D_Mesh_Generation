# taken from https://github.com/hynann/NRDF/blob/master
import sys
sys.path.append('')

import os
import numpy as np
import torch
import glob

from torch.utils.data import Dataset
from pytorch3d.io import load_ply, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import join_meshes_as_batch

# These are pretty much identical, not great for dry code.
dataclass_to_class = {'02691156': 0, '02747177': 1, '02773838': 2, '02801938': 3, '02808440': 4, '02818832': 5, '02828884': 6, '02843684': 7, '02871439': 8, '02876657': 9, '02880940': 10, '02924116': 11, '02933112': 12, '02942699': 13, '02946921': 14, '02954340': 15, '02958343': 16, '02992529': 17, '03001627': 18, '03046257': 19, '03085013': 20, '03207941': 21, '03211117': 22, '03261776': 23, '03325088': 24, '03337140': 25, '03467517': 26, '03513137': 27, '03593526': 28, '03624134': 29, '03636649': 30, '03642806': 31, '03691459': 32, '03710193': 33, '03759954': 34, '03761084': 35, '03790512': 36, '03797390': 37, '03928116': 38, '03938244': 39, '03948459': 40, '03991062': 41, '04004475': 42, '04074963': 43, '04090263': 44, '04099429': 45, '04225987': 46, '04256520': 47, '04330267': 48, '04379243': 49, '04401088': 50, '04460130': 51, '04468005': 52, '04530566': 53, '04554684': 54}


class PointCloudData(Dataset):
    def __init__(self, data_dir, batch_size=4, num_workers=6, num_pts=5000, mode = 'train'):
        self.data_dir = data_dir
        self.num_pts = num_pts

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Implement train / test split !!
        self.mode = mode

        self.data_sequences = sorted(glob.glob(self.data_dir + '/*/*/*/*.ply'))

        self.classes = [dataclass_to_class[i.replace('\\','/').split('/')[-4]] for i in self.data_sequences]

    
    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        verts, faces = load_ply(self.data_sequences[idx])

        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        mesh = Meshes(verts=[verts], faces=[faces])

        sub_sample_mesh = sample_points_from_meshes(mesh, self.num_pts)

        model = {
            'class': self.classes[idx],
            'points': sub_sample_mesh
        }
        
        return model
    
    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn, drop_last=True)
    
    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

class MeshData(Dataset):
    def __init__(self, data_dir, batch_size=4, num_workers=6, mode = 'train'):
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        # Implement train / test split !!
        self.mode = mode

        self.data_sequences = sorted(glob.glob(self.data_dir + '/*/*/*/simplified_mesh.ply'))
        self.classes = [dataclass_to_class[i.replace('\\','/').split('/')[-4]] for i in self.data_sequences]
    
    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        
        # print("Loaded: id: ", idx, " : ", self.data_sequences[idx])
        try:
            verts, faces = load_ply(self.data_sequences[idx])
        except:
            print("Error Occured when loading model with id:", idx, ", path: ",self.data_sequences[idx])
            verts, faces = load_ply(self.data_sequences[0])

        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        mesh = Meshes(verts=[verts], faces=[faces])
        model = {
            'mesh': mesh,
            'class': self.classes[idx]
        }
        
        return model
    

    def collate_fn(self, batch):
        meshes = join_meshes_as_batch([i['mesh'] for i in batch], False)

        model = {
            'meshes': meshes,
            'classes': torch.tensor([int(i['class']) for i in batch])
        }
        return model

    
    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn, collate_fn = self.collate_fn)
    
    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


if __name__ == '__main__':
    data_set = PointCloudData(data_dir = "../ShapeNetCore", batch_size=3, num_pts=5000)
    data_loader = data_set.get_loader()

    print(next(iter(data_loader))['points'].shape)
    
    data_set = MeshData(data_dir = "../ShapeNetCore", batch_size=3)
    data_loader = data_set.get_loader()
    item = next(iter(data_loader))
    print(item['meshes'].verts_packed().shape)
    print(item['classes'])