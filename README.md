# 3D_Mesh_Generation


# IDEA

- Flow matching for 3D mesh generation
- Generate noisy point cloud x and pull meshes from dataset y
- Learn the mapping to turn each point in x to its corresponding point in y
- Make sure to add a smoothness constraint to the genertion (e.g. local neighbourhood smoothness) in order to help generation


# Generating x

1. Generate a random point cloud and assign each point to its closest neighbout in y
    - Problem with this is that there will be multiple points in x that have one nn in y !!

2. Generate x from y by randomly perturbing each point in the normal direction of the vertex for a random number of timesteps (using the normal of its local neighbourhood)
    - There is also the problem of the noise distribution being tied to the meshes !!
    - could be fun to test but will add overhead

3. Generate x from a sphere by randomly perturbing each point in the normal direction of the sphere (using the normal from it's local neighbourhood)


# Training

- We need to generate a curved path that the points in x take in order to get to the mesh y
- 

- If we choose option 2. we can trace the path the node took backwards for each timestep (might be nice)

- If we choose option 1. or 3. we can push the node back towards its corresponding point in y using euclidian distance !!

- It might be worth adding a smoothness constraint on a schedule (e.g. it becomes more important as time goes on)


# TODO

Create a mesh manipulation class based around the Mesh class from torch 3d
    - The class must keep track of vertex, faces, vertex normals
    - be able to manipulte the vertexs and update the faces and vertex normals
    - work with neural networks

[x] download shapnet, 
[x] create class for shapenet input

[ ] create neural network
[ ] Add code for keeping smoothness constraint (Make sure the normal isn't to far from the previous normal!!!)

[ ] add generate x_0 code
[ ] add learning loop, update flowMatching to allow for it to work

[x] add visualisation code


- Torch 3d has padding built in so don't need to worry about masks (except for generation) !!


# Install Requirments

In order to install torch3d visual studio build tools is required (version 19 (16.11.11) is what was used for this project which can be downloaded [here](https://learn.microsoft.com/en-us/visualstudio/releases/2019/history))