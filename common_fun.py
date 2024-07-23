import time
from scipy.sparse import csr_matrix
from dolfin import *
from block import *
import fenics as fe

from xii.assembler.average_matrix import average_matrix as average_3d1d_matrix, trace_3d1d_matrix
from xii import *
from scipy.sparse import *

from ufl.corealg.traversal import traverse_unique_terminals
import dolfin as df
import ufl
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import random


def generate_face_values(min, max, coupling_radius, s):
    random.seed(s)
    
    cube_vertices = [
    (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 0 - 1.1*coupling_radius), (1 + 1.1*coupling_radius, 0- 1.1*coupling_radius, 1 + 1.1*coupling_radius), 
    (1 + 1.1*coupling_radius, 0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius),(0- 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (0 - 1.1*coupling_radius, 1 + 1.1*coupling_radius, 0-1.1*coupling_radius), 
    (0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius, 1 + 1.1*coupling_radius), (0 - 1.1*coupling_radius, 0- 1.1*coupling_radius, 0 - 1.1*coupling_radius)
    ]

    cube_faces = [
    [cube_vertices[0], cube_vertices[1], cube_vertices[3], cube_vertices[2]],  # Faccia 1 (frontale)
    [cube_vertices[4], cube_vertices[5], cube_vertices[7], cube_vertices[6]],  # Faccia 2 (posteriore)
    [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],  # Faccia 3 (superiore)
    [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],  # Faccia 4 (inferiore)
    [cube_vertices[0], cube_vertices[2], cube_vertices[6], cube_vertices[4]],  # Faccia 5 (sinistra)
    [cube_vertices[1], cube_vertices[3], cube_vertices[7], cube_vertices[5]],  # Faccia 6 (destra)
    ]   

    
    face_values = {}
    for i, face in enumerate(cube_faces):
        # Assegna un valore casuale compreso tra 1 e 10 a ciascuna faccia
        rando_value=random.randint(min,max)
        face_values[i] = rando_value*10**-5


    return face_values, cube_vertices, cube_faces

def generate_face_values1(min, max, coupling_radius, s):
    random.seed(s)
    
    cube_vertices = [ (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, -1- 1.1*coupling_radius), (1 + 1.1*coupling_radius, -1- 1.1*coupling_radius, 1 + 1.1*coupling_radius), 
        (1 + 1.1*coupling_radius, -1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius),(-1 - 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (-1- 1.1*coupling_radius, 1 + 1.1*coupling_radius, -1-1.1*coupling_radius), 
        (-1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius, 1 + 1.1*coupling_radius), (-1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius, -1- 1.1*coupling_radius)]


    cube_faces = [
    [cube_vertices[0], cube_vertices[1], cube_vertices[3], cube_vertices[2]],  # Faccia 1 (frontale)
    [cube_vertices[4], cube_vertices[5], cube_vertices[7], cube_vertices[6]],  # Faccia 2 (posteriore)
    [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],  # Faccia 3 (superiore)
    [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],  # Faccia 4 (inferiore)
    [cube_vertices[0], cube_vertices[2], cube_vertices[6], cube_vertices[4]],  # Faccia 5 (sinistra)
    [cube_vertices[1], cube_vertices[3], cube_vertices[7], cube_vertices[5]],  # Faccia 6 (destra)
    ]   

    
    face_values = {}
    for i, face in enumerate(cube_faces):
        # Assegna un valore casuale compreso tra 1 e 10 a ciascuna faccia
        rando_value=random.randint(min,max)
        face_values[i] = rando_value*10**-5


    return face_values, cube_vertices, cube_faces



def compute_vertex_values (cube_vertices, cube_faces, face_values):

    vertex_values_dict={}
    for vertex in cube_vertices:
        adjacent_faces = []
        
        for i, face in enumerate(cube_faces):
            if vertex in face:
                adjacent_faces.append(i)
                
        if not adjacent_faces:
            return None  # Se il vertice non è collegato a nessuna faccia, restituisci None
        
        avg_value = sum(face_values[face_index] for face_index in adjacent_faces) / len(adjacent_faces)
        vertex_values_dict[vertex]=avg_value
    
    return vertex_values_dict

def compute_u(meshV, coupling_radius, vertex_values_dict):

    V = FunctionSpace(meshV, 'CG', 1)
    u=Function(V)
    dof_coordinates = V.tabulate_dof_coordinates()
    dof_indices = V.dofmap().dofs()

    for i in range(len(dof_indices)):
        x, y, z = dof_coordinates[i]
        #print(x,y,z)
        
        # Calcolo delle coordinate normalizzate all'interno dell'elemento cubico
        xi = (x - (-0.011)) / (1.011 - (-0.011))  # Normalizzazione su [0, 1]
        eta = (y - (-0.011)) / (1.011 - (-0.011))
        zeta = (z - (-0.011)) / (1.011 - (-0.011))

        # Coordinate dei vertici circostanti
        vertices = [
        (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 0- 1.1*coupling_radius), (1 + 1.1*coupling_radius, 0- 1.1*coupling_radius, 1 + 1.1*coupling_radius), 
        (1 + 1.1*coupling_radius, 0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius),(0 - 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (0- 1.1*coupling_radius, 1 + 1.1*coupling_radius, 0-1.1*coupling_radius), 
        (0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius, 1 + 1.1*coupling_radius), (0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius, 0- 1.1*coupling_radius)]
        # Valori dei vertici circostanti
        
        vertex_values = [vertex_values_dict[vertex] for vertex in vertices]

        # Valore interpolato sul nodo corrente
        u_values = (vertex_values[7] * (1 - xi) * (1 - eta) * (1 - zeta) +
                    vertex_values[3] * xi * (1-eta) * (1 - zeta) +
                    vertex_values[5] * (1 - xi) * eta * (1 - zeta) +
                    vertex_values[1] * xi * eta * (1 - zeta) +
                    vertex_values[6] * (1 - xi) * (1 - eta) * zeta +
                    vertex_values[2] * xi * (1 - eta) * zeta +
                    vertex_values[4] * (1 - xi) * eta * zeta +
                    vertex_values[0] * xi * eta * zeta)

        # Assegnazione del valore interpolato alla funzione u
        u.vector()[i] = u_values
        


    return u


def compute_u1(meshV, coupling_radius, vertex_values_dict):

    V = FunctionSpace(meshV, 'CG', 1)
    u=Function(V)
    dof_coordinates = V.tabulate_dof_coordinates()
    dof_indices = V.dofmap().dofs()

    for i in range(len(dof_indices)):
        x, y, z = dof_coordinates[i]
        #print(x,y,z)
        
        # Calcolo delle coordinate normalizzate all'interno dell'elemento cubico
        xi = (x - (-1.011)) / (1.011 - (-1.011))  # Normalizzazione su [0, 1]
        eta = (y - (-1.011)) / (1.011 - (-1.011))
        zeta = (z - (-1.011)) / (1.011 - (-1.011))

        # Coordinate dei vertici circostanti
        vertices = [
        (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, -1- 1.1*coupling_radius), (1 + 1.1*coupling_radius, -1- 1.1*coupling_radius, 1 + 1.1*coupling_radius), 
        (1 + 1.1*coupling_radius, -1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius),(-1 - 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius), (-1- 1.1*coupling_radius, 1 + 1.1*coupling_radius, -1-1.1*coupling_radius), 
        (-1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius, 1 + 1.1*coupling_radius), (-1 - 1.1*coupling_radius, -1 - 1.1*coupling_radius, -1- 1.1*coupling_radius)]
        # Valori dei vertici circostanti
        
        vertex_values = [vertex_values_dict[vertex] for vertex in vertices]

        # Valore interpolato sul nodo corrente
        u_values = (vertex_values[7] * (1 - xi) * (1 - eta) * (1 - zeta) +
                    vertex_values[3] * xi * (1-eta) * (1 - zeta) +
                    vertex_values[5] * (1 - xi) * eta * (1 - zeta) +
                    vertex_values[1] * xi * eta * (1 - zeta) +
                    vertex_values[6] * (1 - xi) * (1 - eta) * zeta +
                    vertex_values[2] * xi * (1 - eta) * zeta +
                    vertex_values[4] * (1 - xi) * eta * zeta +
                    vertex_values[0] * xi * eta * zeta)

        # Assegnazione del valore interpolato alla funzione u
        u.vector()[i] = u_values
        


    return u


def print_u (u, meshV):

    u_values = u.compute_vertex_values(meshV)

    # Visualizzazione tridimensionale della funzione u
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Traccia i punti della mesh con i valori della funzione u
    scatter=ax.scatter(meshV.coordinates()[:, 0], meshV.coordinates()[:, 1], meshV.coordinates()[:, 2], c=u_values, cmap='viridis')
    colorbar = fig.colorbar(scatter)

    # Etichette degli assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()    


def project_solution(submesh, u3d):
    W=FunctionSpace(submesh, 'CG', 1)
    u_proj=Function(W)
    u_proj=interpolate(u3d, W)
    
    return u_proj    