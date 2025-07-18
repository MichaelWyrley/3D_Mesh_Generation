import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.transforms as transforms
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.loss import (
    chamfer_distance, 
)
from losses import (mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,)
from pytorch3d.structures import Meshes
from tqdm.notebook import tqdm
from pytorch3d.ops import sample_points_from_meshes
from math import ceil

class FlowMatching():

    def __init__(self, model, training_type=3, device='cuda', simulation_iterations=150, simulation_lr=15.0, simulation_detail = 5000, sphere_detail=4, x_0_purtubation_amount=20, x_0_purtubation_power=0.01):
        """
        :param model: the neural network used for vector field predictions
        :param device: the torch device it will be run on
        :training_type: the type of training to be done (1,2,3)
        """

        self.model = model

        self.device = device
        self.training_type = training_type
        self.device = device

        self.sigma_small = torch.tensor([0.0001], device=self.device)
        self.epsilon = 0.000001
        self.simulation_iterations = simulation_iterations
        self.simulation_lr = simulation_lr
        self.simulation_detail = simulation_detail

        self.sphere_detail = sphere_detail
        self.x_0_purtubation_amount = x_0_purtubation_amount
        self.x_0_purtubation_power = x_0_purtubation_power
        

    def sample_timestep(self, n):
        """
        :param n: number of timesteps needed to be generated
        """

        return torch.rand(n, device=self.device).unsqueeze(-1)
    

    def random_generation(self, bs):

        x_0_mesh = ico_sphere(self.sphere_detail, self.device).extend(bs)
        verts, faces = x_0_mesh.verts_padded(), x_0_mesh.faces_padded()

        x_0_verts = torch.normal(mean = (0,0,0), std = (1,1,1), size=verts.shape)

        x_0 = Meshes(verts = x_0_verts, faces = faces)

        
        raise Exception("Not currently implemented!!")
        # return torch.randn_like(x_1)

    def purturbed_x_1(self, x_1):
        x_0_mesh = x_1.extend(1)

         # Diform the vertexes by the normals of them, in order to get a good representation !!
        for i in range(self.x_0_purtubation_amount):
            normals = x_0_mesh.verts_normals_packed()

            deform_offset = normals[0] * (torch.rand((normals.shape[0],1), device = x_0_mesh.device) * 2. -1) * self.x_0_purtubation_power

            # This might need to be multiplied, make sure to debug!!
            x_0_mesh = x_0_mesh.offset_verts(deform_offset)
        
        return x_0_mesh

    def purturbed_sphere(self, bs):
        x_0_mesh = ico_sphere(self.sphere_detail, self.device).extend(bs)

        # Diform the vertexes by the normals of them, in order to get a good representation !!
        for i in range(self.x_0_purtubation_amount):
            normals = x_0_mesh.verts_normals_packed()

            deform_offset = normals[0] * (torch.rand((normals.shape[0],1), device = x_0_mesh.device) * 2. -1) * self.x_0_purtubation_power

            # This might need to be multiplied, make sure to debug!!
            x_0_mesh = x_0_mesh.offset_verts(deform_offset)

        return x_0_mesh
    
    def generate_x_0(self, x_1):
        if self.training_type == 1:
            bs = x_1.verts_padded().shape[0]
            return self.random_generation(bs)
        if self.training_type == 2:
            # generate x_0 by simulating the movement of x_1 along it's normals for a random number of timesteps
            return self.purturbed_x_1(x_1)
        if self.training_type == 3:
            bs = x_1.verts_padded().shape[0]
            return self.purturbed_sphere(bs)
        else:
            raise Exception("training type not in allowed list [1,2,3]")        

    # simulate the path that the points travel on to move from x_0 to x_1 using chamfer distance optimisation!
    # Code modified from https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb
    def simulate_path(self, x_0, x_1, t):
        deform_verts = [torch.full(i.verts_packed().shape, 0.0, device=self.device, requires_grad=True) for i in x_0]
        optimizer = torch.optim.SGD(deform_verts, lr=self.simulation_lr, momentum=0.9)


        w_chamfer = 1.0 
        w_edge = 1.0 
        w_normal = 0.01 
        w_laplacian = 0.1 
        
        max_iter = int(t.max())+1
        simulation = [{'mesh': x_0.extend(1), 'vec': torch.cat(deform_verts, dim = 0).clone(), 'dist': torch.tensor([999999.,99999.,99999.])}]

        for i in range(max_iter):
            # print("simulation iteration: ", i)
            optimizer.zero_grad()
            
            # Deform the mesh
            new_src_mesh = x_0.offset_verts(torch.cat(deform_verts))

            sample_trg = sample_points_from_meshes(x_1, self.simulation_detail)
            sample_src = sample_points_from_meshes(new_src_mesh, self.simulation_detail)

            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src, batch_reduction=None)
            loss_edge = mesh_edge_loss(new_src_mesh)
            loss_normal = mesh_normal_consistency(new_src_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

            simulation.append({'mesh': new_src_mesh.extend(1), 'vec': torch.stack(deform_verts, dim = 0).clone(), 'dist': loss_chamfer.clone()})

            loss = loss.mean()
            # Optimization step
            loss.backward()
            optimizer.step()

        return simulation
        
    
    def conditional_flow(self, x_0, x_1, t):
        amount_to_simulate = (self.simulation_iterations * t).ceil().int()
        
        simulation = self.simulate_path(x_0.clone(), x_1, amount_to_simulate)

        idxes = {'mesh': [], 'vec': [], 'dist': []}
        for i in range(len(amount_to_simulate)):
            idxes['mesh'].append(simulation[amount_to_simulate[i]]['mesh'][i])
            idxes['vec'].append(simulation[amount_to_simulate[i]]['vec'][i])
            idxes['dist'].append(simulation[amount_to_simulate[i]]['dist'][i])
        
        idxes['mesh'] = join_meshes_as_batch(idxes['mesh'])
        idxes['vec'] = torch.stack(idxes['vec'], dim = 0)
        idxes['dist'] = torch.stack(idxes['dist'], dim = 0)

        return idxes
    
    def conditional_vector_field(self, x_0, x_t, x_1, epsilon=0.00001):
        sample_x_1 = sample_points_from_meshes(x_1, self.simulation_detail)
        sample_x_0 = sample_points_from_meshes(x_0, self.simulation_detail)

        # USE THE GRADIENT OF 'vec' !!!

        d_0_1, _ = chamfer_distance(sample_x_0, sample_x_1, batch_reduction=None)
        grad_d_t_1 = x_t['vec']

        out = d_0_1[:,None,None] * grad_d_t_1 / ((torch.linalg.norm(grad_d_t_1, dim=(2))+epsilon).unsqueeze(-1))

        if (out.isnan().any()):
            out = torch.nan_to_num(out, nan=0.0)

        return out

    def apply_nn(self, x, t, c = None):


        verts = x.verts_padded() 
        v_t = self.model(verts, t, c)

        return v_t

    def train_step(self, x_1, c = None):

        x_0 = self.generate_x_0(x_1)

        t = self.sample_timestep(x_1.verts_padded().shape[0]).requires_grad_(True)

        psi_t = self.conditional_flow(x_0, x_1, t)
        v_t = self.apply_nn(psi_t['mesh'], t, c)

        con_vec = self.conditional_vector_field(x_0, psi_t, x_1)

        loss = F.mse_loss(v_t, con_vec)

        return loss
    
    
    def sample(self, n, timesteps=50, scale=1, labels=None):
        self.model.eval()
        with torch.no_grad():
            z = self.purturbed_sphere(n).to(self.device)

            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)

            for i in range(timesteps):
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.apply_nn(z, t, labels)

                z = z.offset_verts(v_t.reshape(-1, 3))
                
        self.model.train()

        return z
    

def test_main_one_mesh():
    from visualiser import visualise_mesh
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.io import load_obj

    from neural_network import DiT_adaLN_zero

    device = 'cuda'
    
    neural_net = DiT_adaLN_zero(in_dim=3, depth=2, emb_dimention=18, num_heads=3, num_classes=55, device = device)
    model = FlowMatching(neural_net, device = device)

    verts, faces, _ = load_obj("../ShapeNetCore/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj")

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    x_1 = Meshes(verts=[verts], faces=[faces.verts_idx]).to(device = device)
    
    save_dir = './outputs/images/test'

    x_0 = model.generate_x_0(x_1)
    
    # visualise_mesh(x_1,save_dir + '_base_x1_mesh')
    # visualise_mesh(x_0,save_dir + '_base_x0_mesh')

    # t = model.sample_timestep(x_1.verts_padded().shape[0])
    t = torch.ones(x_1.verts_padded().shape[0], device = device).unsqueeze(-1)
    amount_to_simulate = (model.simulation_iterations * t).ceil().int()

    x_0_simulated = model.simulate_path(x_0, x_1, amount_to_simulate)
    # visualise_mesh(x_0_simulated[-1]['mesh'],save_dir + '_simulated_x0_mesh')

    # for i in range(0,len(x_0_simulated), 10):
    #     print('Outputing: ', i)
    #     visualise_mesh(x_0_simulated[i],"./outputs/images/simulation/{:03d}".format(i)) 
    # 
    #    
    psi_t = x_0_simulated[amount_to_simulate]
    # print(x_1.verts_padded().shape, psi_t.verts_padded().shape, psi_t.verts_padded().device)
    v_t = model.apply_nn(psi_t['mesh'], t, torch.tensor(0).to(device))

    con_vec = model.conditional_vector_field(x_0, psi_t, x_1)

    print(con_vec.shape, v_t.shape)

    loss = F.mse_loss(v_t, con_vec)

def test_main_multiple_meshes():
    from visualiser import visualise_mesh
    from dataset import MeshData

    from neural_network import DiT_adaLN_zero

    device = 'cuda'
    save_dir = './outputs/images/test'

    neural_net = DiT_adaLN_zero(in_dim=3, depth=2, emb_dimention=18, num_heads=3, num_classes=55, device = device, condition_prob=1.0)
    model = FlowMatching(neural_net, device = device, simulation_iterations=300, simulation_lr=12, simulation_detail = 7500, sphere_detail=4, x_0_purtubation_amount=20, x_0_purtubation_power=0.01)

    data_set = MeshData(data_dir = "../ShapeNetCore", batch_size=4)
    data_loader = data_set.get_loader()
    item = next(iter(data_loader))
    
    x_1 = item['meshes'].to(device)
    x_1_class = item['classes'].to(device)
    x_0 = model.generate_x_0(x_1)

    visualise_mesh(x_1,save_dir + '_base_x1_mesh')
    visualise_mesh(x_0,save_dir + '_base_x0_mesh')

    # t = model.sample_timestep(x_1.verts_padded().shape[0])
    t = torch.ones(x_1.verts_padded().shape[0], device = device).unsqueeze(-1)
    amount_to_simulate = (model.simulation_iterations * t).ceil().int()

    psi_t = model.conditional_flow(x_0, x_1, t)
    visualise_mesh(psi_t['mesh'],save_dir + '_psi_t')

    # simulation = []
    
    import time
    start = time.time()
    print("start")
    simulated_path = model.simulate_path(x_0, x_1, amount_to_simulate)
    print("end, {:.3f}".format(time.time() - start))

    # print(amount_to_simulate[0], ", ", len(simulation))
    for i in range(0,amount_to_simulate[0], 10):
        visualise_mesh(simulated_path[i]['mesh'], "./outputs/images/simulation/{:03d}".format(i))
    
    v_t = model.apply_nn(psi_t['mesh'], t, x_1_class)

    con_vec = model.conditional_vector_field(x_0, psi_t, x_1)

    print(con_vec.shape, v_t.shape)

    loss = F.mse_loss(v_t, con_vec)
    
def test_sample():
    from visualiser import visualise_mesh
    from dataset import MeshData

    from neural_network import DiT_adaLN_zero

    device = 'cuda'
    save_dir = './outputs/images/test'

    neural_net = DiT_adaLN_zero(in_dim=3, depth=2, emb_dimention=18, num_heads=3, num_classes=55, device = device, condition_prob=1.0)
    model = FlowMatching(neural_net, device = device, simulation_iterations=300, simulation_lr=15.0, simulation_detail = 5000, sphere_detail=4, x_0_purtubation_amount=20, x_0_purtubation_power=0.01)

    sample = model.sample(3)

    visualise_mesh(sample,save_dir + '_sample')

if __name__ == '__main__':
    # test_main_one_mesh()
    test_main_multiple_meshes()
    # test_sample()