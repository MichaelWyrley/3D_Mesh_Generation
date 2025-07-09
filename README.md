# 3D_Mesh_Generation


# IDEA

- Flow matching for 3D mesh generation
- Generate a noisy sphere and transform it onto a 3D mesh by minimising Chamfer Distance 
- We can learn this transformation using flow matching to generate new 3D meshes

# Generating x

1. Generate a random point cloud and assign each point to its closest neighbout in y
    - Problem with this is that there will be multiple points in x that have one nn in y !!

2. Generate x from y by randomly perturbing each point in the normal direction of the vertex for a random number of timesteps (using the normal of its local neighbourhood)
    - There is also the problem of the noise distribution being tied to the meshes !!
    - could be fun to test but will add overhead

3. Generate x from a sphere by randomly perturbing each point in the normal direction of the sphere (using the normal from it's local neighbourhood)


# Training

- Import Mesh from [ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore) 


# TODO


[x] download shapnet, 
[x] create class for shapenet input

[x] create neural network
[x] Add code for keeping smoothness constraint (Make sure the normal isn't to far from the previous normal!!!)

[x] add generate x_0 code
[x] add learning loop, update flowMatching to allow for it to work

[x] add visualisation code




# Install Requirments

Install Torch 2.4.1

In order to install torch3d visual studio build tools is required (version 19 (16.11.11) is what was used for this project which can be downloaded [here](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history))

# Acknowledgement

```
@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}
```

```
@techreport{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel X. and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  number      = {arXiv:1512.03012 [cs.GR]},
  institution = {Stanford University --- Princeton University --- Toyota Technological Institute at Chicago},
  year        = {2015}
}
```
