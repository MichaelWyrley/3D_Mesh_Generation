import sys
sys.path.append('')
import torch
import os
import copy
import numpy as np

from flowMatching import FlowMatching
from neural_network import DiT_adaLN_zero

from pytorch3d.io import save_obj

from dataset import MeshData
from visualiser import visualise_mesh
from tqdm import tqdm


def train(args):

    model = DiT_adaLN_zero().to(args['device'])

    flow = FlowMatching(model,training_type= args['training_type'], device=args['device'], simulation_detail=args['simulation_detail'], sphere_detail = args['sphere_detail'])


    data_set = MeshData(data_dir = args['dataset_dir'], batch_size=args['batch_size'], num_workers=0)
    data_loader = data_set.get_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    print("Starting Training")
    for epoch in range(0, args['epochs']+1):
        loop = tqdm(data_loader)
        c = 1
        for data in loop:
            c+=1
            optimizer.zero_grad()
            meshes, classes = data['meshes'].to(args['device']), data['classes'].to(args['device'])

            loss = flow.train_step(meshes, classes)

            del meshes
            del classes

            loop.set_description('epoch = {}, loss = {:.5f}'.format(epoch, loss))

            # if c % 2000 == 0:
            #     torch.save(model.state_dict(), f"{args['save_dir']}/models/model_c_{c}")
            #     sampled = flow.sample(args['samples'], scale=1)

            #     visualise_mesh(sampled, f"{args['save_dir']}/sample_generation/epoch_c_{c}", args['save_grid'])

            loss.backward()
            optimizer.step()

        if epoch % args['output_epocs'] == 0:
            torch.save(model.state_dict(), f"{args['save_dir']}/models/model_{epoch}")
            sampled = flow.sample(args['samples'], scale=1)

            visualise_mesh(sampled, f"{args['save_dir']}/sample_generation/epoch_{epoch}", args['save_grid'])


    torch.save(model.state_dict(), f"{args['save_dir']}/models/model_final")
    sampled = flow.sample(args['samples'], scale=1).cpu().numpy()
    visualise_mesh(sampled, f"{args['save_dir']}/sample_generation/final", args['save_grid'])



if __name__ == '__main__':

    args = {
        'dataset_dir': "../ShapeNetCore",
        'save_dir': 'outputs',

        'batch_size': 7,
        'simulation_detail': 5000,
        'lr': 1e-4,
        'epochs': 400,
        'sphere_detail': 1,

        'output_epocs': 20,
        'samples': 16,
        'save_grid': True,

        'device': 'cuda',

        'training_type': 3,

    }
    
    # os.mkdir('./samples', exist_ok=True)
    # os.mkdir('./samples/training_models', exist_ok=True)
    # os.mkdir('./samples/training_samples', exist_ok=True)

    train(args)