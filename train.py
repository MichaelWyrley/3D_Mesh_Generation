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

    flow = FlowMatching(model, training_type= args['training_type'], device=args['device'])


    data_set = MeshData(data_dir = args['dataset_dir'], batch_size=args['batch_size'])
    data_loader = data_set.get_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    print("Starting Training")
    for epoch in range(0, args['epochs']+1):
        loop = tqdm(data_loader)
        for data in loop:
            optimizer.zero_grad()

            meshes, classes = data['meshes'].to(args['device']), data['classes'].to(args['device'])

            loss = flow.train_step(meshes, classes)

            loop.set_description('epoch = {}, loss = {:.5f}'.format(epoch, loss))

            loss.backward()
            optimizer.step()

        if epoch % 25 == 0:
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

        'batch_size': 5,
        'generation_quality': 5000,
        'lr': 1e-4,
        'epochs': 100,

        'samples': 16,
        'save_grid': True,

        'device': 'cuda',

        'training_type': 3,

    }
    
    # os.mkdir('./samples', exist_ok=True)
    # os.mkdir('./samples/training_models', exist_ok=True)
    # os.mkdir('./samples/training_samples', exist_ok=True)

    train(args)