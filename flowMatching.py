import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch3d.transforms as transforms
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from tqdm.notebook import tqdm
from pytorch3d.ops import sample_points_from_meshes
from math import ceil

class FlowMatching():

    def __init__(self, model, training_type=3, device='cuda', simulation_iterations=300, simulation_lr=8.0, simulation_detail = 10000, sphere_detail=50, x_0_purtubation_amount=20, x_0_purtubation_power=0.05):
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
    

    def random_generation(self, x_1):
        raise Exception("Sorry, not implemented random generation yet") 
        # return torch.randn_like(x_1)

    def purturbed_x_1(self, x_1):
        x_0_mesh = x_1.extend(1)

         # Diform the vertexes by the normals of them, in order to get a good representation !!
        for i in range(self.x_0_purtubation_amount):
            normals = x_0_mesh.verts_normals_padded()

            deform_offset = normals[0] * (torch.rand((normals.shape[1],1), device = x_0_mesh.device) * 2. -1) * self.x_0_purtubation_power

            # This might need to be multiplied, make sure to debug!!
            x_0_mesh = x_0_mesh.offset_verts(deform_offset)
        
        return x_0_mesh

    def purturbed_sphere(self, x_1):
        x_0_mesh = ico_sphere(ceil(self.sphere_detail ** (1/4)), self.device).extend(x_1.verts_padded().shape[0])

        # Diform the vertexes by the normals of them, in order to get a good representation !!
        for i in range(self.x_0_purtubation_amount):
            normals = x_0_mesh.verts_normals_padded()

            deform_offset = normals[0] * (torch.rand((normals.shape[1],1), device = x_0_mesh.device) * 2. -1) * self.x_0_purtubation_power

            # This might need to be multiplied, make sure to debug!!
            x_0_mesh = x_0_mesh.offset_verts(deform_offset)

        return x_0_mesh
    
    def generate_x_0(self, x_1):
        if self.training_type == 1:
            return self.random_generation(x_1)
        if self.training_type == 2:
            # generate x_0 by simulating the movement of x_1 along it's normals for a random number of timesteps
            return self.purturbed_x_1(x_1)
        if self.training_type == 3:
            return self.purturbed_sphere(x_1)
        else:
            raise Exception("training type not in allowed list [1,2,3]")
        
    # simulate the path that the points travel on to move from x_0 to x_1 using chamfer distance optimisation!
    # Code modified from https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/deform_source_mesh_to_target_mesh.ipynb
    def simulate_path(self, x_0, x_1, t):
        deform_verts = torch.full(x_0.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
    
        w_chamfer = 1.0 
        w_edge = 1.0 
        w_normal = 0.01 
        w_laplacian = 0.1 

        
        max_iter = int(t.max())+1
        simulation = [x_0.extend(1)]

        print('simulating trajectory')
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # Deform the mesh
            new_src_mesh = x_0.offset_verts(deform_verts)
            simulation.append(new_src_mesh.extend(1))

            sample_trg = sample_points_from_meshes(x_1, self.simulation_detail)
            sample_src = sample_points_from_meshes(new_src_mesh, self.simulation_detail)

            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
            loss_edge = mesh_edge_loss(new_src_mesh)
            loss_normal = mesh_normal_consistency(new_src_mesh)
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
            
            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

            # Optimization step
            loss.backward()
            optimizer.step()
            
        # return simulation[t]
        return simulation
        
    
    def conditional_flow(self, x_0, x_1, t):
        amount_to_simulate = (self.simulation_iterations * t).ceil().int()
        return self.simulate_path(x_0, x_1, amount_to_simulate)

        # return (1 - (1 - self.sigma_small) * t) * x_0 + t * x_1
    
    # USE THE CODE FROM Pose-RFM to get the gradient of the chamfer distance !!
    # WILL NEED TO CHANGE conditional flow to also use the minimum gradient !!
    def conditional_vector_field(self, x_0, x_t, x_1, t, epsilon=0.00001):
        d_0_1 = chamfer_distance(x_0, x_1)
        d_t_1 = chamfer_distance(x_t, x_1)

        grad_d_t_1 = torch.autograd.grad(
            inputs=x_t,
            outputs=d_t_1,
            grad_outputs=torch.ones_like(d_0_1),
            create_graph=True,
            retain_graph=True)[0]


        out = d_0_1[:, :,None,None] * grad_d_t_1 / ((torch.linalg.norm(grad_d_t_1, dim=(2,3))+epsilon)[:, :,None,None])
        
        if (out.isnan().any()):
            out = torch.nan_to_num(out, nan=0.0)

        return out
        # return (x_1 - (1 - self.sigma_small) * x_0)

    def apply_nn(self, x, t, mask, c = None):

        # TODO: Deal with mask
        v_t = self.model(x, t, c)

        return v_t

    def train_step(self, x_1, c = None):

        x_0 = self.generate_x_0(self, x_1)

        t = self.sample_timestep(x_1.verts_padded().shape[0]).requires_grad_(True)

        # Maybe try to learn the chamfer distance gradient instead of the euclidian vector !!

        psi_t = self.conditional_flow(x_0, x_1, t[:, None])
        v_t = self.apply_nn(psi_t, t, c)

        con_vec = self.conditional_vector_field(x_0, psi_t, x_1, t)

        loss = F.mse_loss(v_t, con_vec)

        return loss
    
    def sample(self, n, timesteps=50, scale=1, labels=None):
        x_0_mesh = ico_sphere(2, self.device).extend(n)

        self.model.eval()
        with torch.no_grad():
            z = self.generate_x_0(x_0_mesh).to(self.device)

            steps = torch.linspace(0.0, 1.0, timesteps, device=self.device)
            mask = torch.ones(z)

            for i in range(timesteps):
                t = torch.full((n,1), steps[i], device=self.device)
                v_t = self.apply_nn(z,t, mask, labels)

                z = z + 1 / timesteps * v_t * scale
                
        self.model.train()

        return z
    

if __name__ == '__main__':
    from visualiser import visualise_mesh
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    
    model = FlowMatching(None, device = 'cuda')

    verts, faces, _ = load_obj("../ShapeNetCore/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj")

    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    x_1 = Meshes(verts=[verts], faces=[faces.verts_idx]).to(device = 'cuda')
    
    save_dir = './outputs/images/test'

    x_0 = model.generate_x_0(x_1)
    
    visualise_mesh(x_1,save_dir + '_base_x1_mesh')
    visualise_mesh(x_0,save_dir + '_base_x0_mesh')

    # t = model.sample_timestep(x_1.verts_padded().shape[0])
    t = torch.ones(x_1.verts_padded().shape[0], device = 'cuda').unsqueeze(-1)

    x_0_simulated = model.conditional_flow(x_0, x_1, t)
    # print(x_0_simulated.shape)

    visualise_mesh(x_0_simulated[-1],save_dir + '_simulated_x0_mesh')    

    for i in range(0,len(x_0_simulated), 10):
        print('Outputing: ', i)
        visualise_mesh(x_0_simulated[i],"./outputs/images/simulation/{:03d}".format(i))    

    

    