# Introduction and Code Structure
This repository can be used to implement the Domain Decomposition Reduced Order Model algorithm applied to a coupled tissue-capillary problem, to compute the pressure of the blood flow.

In particular, it is structured as follows:

- [common_fun.py](./common_fun.py), [closest_point_in_mesh.py](./closest_point_in_mesh.py): these Python files group all the auxiliary functions used to infer and analyze this problem
- [podminn.py](./podminn.py), [np2vtk.py](./np2vtk.py): these Pyhton files contain auxiliary function used to set up the Reduced Order Model.
- [rom_training.ipynb](https://github.com/violabrenzone/DD-ROM/blob/main/rom_training.ipynb): Jupyter notebook that contains the configuration and traininig of the Reduced Order Model
- [fom_loc_def.py](./fom_loc_def.py): this Python file is used to generate the snapshots needed to create the training dataset.


# Prerequitsites

- Python 3.x
- Required Python Libraries:  `NumPy`, `SciPy`, `dolfin`, `fenics`, `petsc4py`,`block`, `xii`, `matplotlib`,`dlroms`
