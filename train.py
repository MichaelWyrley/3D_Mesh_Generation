import sys
sys.path.append('')
import torch
import os
import copy
import numpy as np

from flowMatching import FlowMatching
from neural_network import MeshTransformer

from pytorch3d.io import save_obj

from dataset import MeshData
from visualiser import visualise


def train(args):

    model = MeshTransformer().to(args['device'])

    flow = FlowMatching(model, training_type= args['training_type'], device=args['device'])


    data_set = MeshData(data_dir = args['dataset_dir'], batch_size=args['batch_size'], num_pts=args['generation_quality'], mode='train')
    data_loader = data_set.get_loader()

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    print("Starting Training")
    for epoch in range(0, args['epochs']+1):
        for i, data in enumerate(data_loader):
            optimizer.zero_grad()

            meshes, classes = data['meshes'], data['classes']

            loss = flow.train_step(meshes, classes)

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item()}")

        print(f"Epoch [{epoch}] Loss: {loss.item()}")

        if epoch % 25 == 0:
            torch.save(model.state_dict(), f"{args['save_dir']}/models/model_{epoch}")
            sampled = flow.sample(args['samples'], scale=1)

            visualise(sampled, f"{args['save_dir']}/images/epoch_{epoch}", args['save_grid'])


    torch.save(model.state_dict(), f"{args['save_dir']}/models/model_final")
    sampled = flow.sample(args['samples'], scale=1).cpu().numpy()
    visualise(sampled, f"{args['save_dir']}/images/final", args['save_grid'])



if __name__ == '__main__':

    args = {
        'dataset_dir': 'dataset/',
        'save_dir': 'outputs',

        'batch_size': 20,
        'generation_quality': 5000,
        'lr': 1e-4,
        'epochs': 100,

        'samples': 16,
        'save_grid': True,

        'device': 'cuda',

        'training_type': 1,

    }
    
    # os.mkdir('./samples', exist_ok=True)
    # os.mkdir('./samples/training_models', exist_ok=True)
    # os.mkdir('./samples/training_samples', exist_ok=True)

    train(args)