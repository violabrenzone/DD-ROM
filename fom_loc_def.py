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
import common_fun as fun


def get_mesh(n, coupling_radius, path_to_1Dmesh):

    #--------------------MESHES-------------------------

    #3D mesh--------------------------------------------
    inf_point = Point(0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius, 0 - 1.1*coupling_radius)
    max_point = Point(1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius, 1 + 1.1*coupling_radius)
    meshV = BoxMesh(inf_point, max_point, n, n, n)
    coord_V=meshV.coordinates()
    #--------------------------------------------------
    
    #marking 3D mesh facets

    class inlet1(SubDomain):
         def inside(self, x, on_boundary):
             return near(x[0], -1.11) and on_boundary

    class inlet2(SubDomain):
         def inside(self, x, on_boundary):
             return near(x[1], -1.11) and on_boundary

    class inlet3(SubDomain):
         def inside(self, x, on_boundary):
             return near(x[2], -1.11) and on_boundary



    class outlet1(SubDomain):
         def inside(self, x, on_boundary):
             return near(x[0], 1.11) and on_boundary

    class outlet2(SubDomain):
         def inside(self, x, on_boundary):
             return near(x[1], 1.11) and on_boundary

    class outlet3(SubDomain):
         def inside(self, x, on_boundary):
             return near(x[2], 1.11) and on_boundary




    V_markers = MeshFunction('size_t', meshV, meshV.topology().dim()-1) #facet markers

    inlet1().mark(V_markers, 111)
    inlet2().mark(V_markers, 111)
    inlet3().mark(V_markers, 111)

    outlet1().mark(V_markers, 999)
    outlet2().mark(V_markers, 999)
    outlet3().mark(V_markers, 999)
 

    #1D mesh
    
    #LOADING NETWORK XDMF 1D MESH AND ITS TAGS (each mesh function must be saved in different files)
    meshQ = Mesh()
    print(f'{path_to_1Dmesh}.xdmf')
    with XDMFFile(f'{path_to_1Dmesh}marked_mesh.xdmf') as infile:
        infile.read(meshQ)
    print(f'{path_to_1Dmesh}markers.xdmf') 
    Q_markers = MeshFunction('size_t', meshQ , 0)
  
    xdmf_file = XDMFFile(f'{path_to_1Dmesh}markers.xdmf')  
    
    xdmf_file.read(Q_markers)
    xdmf_file.close()
    
    #---------------------------------------------------------------
        
    #marking
    
    tag = 111
    tag_out=999 # per input. 999 per output.
    
    ds1d = Measure('ds', domain=meshQ)
    ds1d = ds(subdomain_data=Q_markers)

    '''
    ds3d = Measure('ds', domain=meshV)
    ds3d = ds3d(subdomain_data=V_markers)
    '''

    return meshV, meshQ, Q_markers, ds1d


def get_system(meshV, meshQ, ubc3 , ubc1, ds1d,coupling_radius=0.001):
    """A, b, W, bcs"""

    Q = FunctionSpace(meshQ, 'CG', 1)
    Q1 = FunctionSpace(meshV, 'CG', 1)

    dofmap = Q1.dofmap()
    V_DOF = dofmap.global_dimension()
    print("ecco 3D dofs", V_DOF)

    W = [Q1, Q]

    u, p = map(TrialFunction, W)
    v, q = map(TestFunction, W)
   
    #computing average operator------------------------------------------
   
    # Average (coupling_radius > 0) or trace (coupling_radius = 0)
    if coupling_radius > 0:
        # Averaging surface
        cylinder = Circle(radius=coupling_radius, degree=10)
        Ru, Rv = Average(u, meshQ, cylinder), Average(v, meshQ, cylinder)
        C = average_3d1d_matrix(Q1, Q, cylinder)
    else:
        Ru, Rv = Average(u, meshQ, None), Average(v, meshQ, None)
        C = trace_3d1d_matrix(Q1, Q, meshQ)

    #--------------------------------------------------------------------         
        
    # Line integral
    dx_ = Measure('dx', domain=meshQ)

    #-------------------------------------------------------------------

    # We're building a 2x2 problem with 3D reaction---------------------
    
    R=coupling_radius

    # mm - bar
    K1=Constant(7.6*10**-11)
    K=Constant(1.03*10**-6/(8*5*10**-4)* 3.14 * R**4*10**5)
    J=Constant(6.28*10**-12)
    #B=Constant(10**-6)
    B=Constant(10**-9)
    P=Constant((3300-666)*10**-5)
    lamb=1
    lamb=Constant(-lamb*np.pi*R**2)
    
    # Boundary Conditions
    p3d_exact=Constant(0.04)

    p1d_in = Constant(0.07) #boundary value
    p1d_out=Constant(0.05)
    tag_in=111
    tag_out=999
    a = block_form(W, 2)
    a[0][0] = K1 * inner(grad(u), grad(v)) * dx + B * inner(u, v) * ds
    #a[0][0] = K1 * inner(grad(u), grad(v)) * dx + B * inner(u, v) * ds3d(111, domain=meshV) +  B * inner(u, v) * ds3d(999, domain=meshV)
    a[1][1] = K * inner(grad(p), grad(q)) * dx -lamb*inner(p,q)*ds1d(tag_in,domain=meshQ)-lamb*inner(p,q)*ds1d(tag_out,domain=meshQ)
    
    m = block_form(W, 2)
    m[0][0] =  J* inner(Ru, Rv) * dx_
    m[0][1] = -J* inner(p, Rv) * dx_
    m[1][0] = -J*inner(q, Ru) * dx_
    m[1][1] = J*inner(p, q) * dx_

    L = block_form(W, 1)
    #L[0] =  - J*inner(P,Rv) * dx_ + B*inner(p3d_exact,v)*ds3d(111,domain=meshV)
    #L[0] =  - J*inner(P,Rv) * dx_ + B*inner(ubc,v)*ds3d(111, domain=meshV) + B*inner(ubc,v)*ds3d(999, domain=meshV)
    L[0] =  - J*inner(P,Rv) * dx_ + B*inner(ubc3,v)*ds

    L[1] =  + J*inner(P,q)*dx_ - lamb*inner(ubc1,q)*ds1d(tag_in,domain=meshQ)- lamb*inner(ubc1,q)*ds1d(tag_out,domain=meshQ)
    
    print(np.shape(a))
    print(np.shape(m))
    print(np.shape(L))

    AD, M, b = map(ii_assemble, (a, m, L))
    AAA = ii_assemble(a + m)
    
    C = csr_matrix(C.getValuesCSR()[::-1], shape=C.size)


    return (AD, M), b, W, C, V_DOF, AAA

def boundary_cond3d(min, max, meshV,seed):
    
    
    coupling_radius=1e-2
    face_values, cube_vertices, cube_faces=fun.generate_face_values(min,max,coupling_radius, seed)
    vertex_values_dict=fun.compute_vertex_values(cube_vertices, cube_faces, face_values)
    u_d=fun.compute_u(meshV, coupling_radius, vertex_values_dict)   
    #fun.print_u(u_d, meshV)

    return u_d

def val_facce(min, max, meshV, seed):
    coupling_radius=1e-2
    face_values, cube_vertices, cube_faces=fun.generate_face_values(min,max,coupling_radius, seed)
    val_list=list(face_values.values())
    val=np.array(val_list)  
    return val



def boundary_cond1d(min, max, meshQ, meshV, seed):

    u_3d=boundary_cond3d(min,max,meshV,seed)
    Q=FunctionSpace(meshQ, 'CG',1)
    
    u_1d=Function(Q)
    u_1d=interpolate(u_3d, Q)

    return u_1d

def getGraphDist(mesh1D, mesh3D, u_dict):
   
    from closest_point_in_mesh import  closest_point_in_mesh
    '''Getting the  distance function from a mesh'''
    
    dist = MeshFunction("double", mesh3D,0)
    coord1d=mesh1D.coordinates()
    
    tree=mesh1D.bounding_box_tree()
    tree.build(mesh1D,1)
    
    for i, cord in enumerate(mesh3D.coordinates()):
    
        close_p = closest_point_in_mesh(cord, mesh1D)
        p=tuple(close_p)
        
        if p in u_dict:
            u=u_dict[p]
            val=(1-np.linalg.norm(cord-close_p))*u
        else:
            point=Point(close_p[0] , close_p[1], close_p[2])
            cell_ind,_=tree.compute_closest_entity(point)
            m=MeshEntity(mesh1D, 1,cell_ind)
            line=m.entities(0)
            p1=tuple(coord1d[line[0]])
            u1=u_dict[p1]
            p2=tuple(coord1d[line[1]])
            u2=u_dict[p2]
            d1=np.linalg.norm(close_p-coord1d[line[0]])
            d2=np.linalg.norm(close_p-coord1d[line[1]])
            d=d1+d2
            u=u1*d1/d+u2*d2/d
            val=(1-np.linalg.norm(cord-close_p))*u
            
        dist.array()[i] = val
       
    return dist

def u_surf(u3d,meshV):
    boundary_mesh = BoundaryMesh(meshV, 'exterior')
    V_surface = FunctionSpace(boundary_mesh, 'CG', 1)
    usurf=Function(V_surface)
    usurf=interpolate(u3d, V_surface)
    coordext=V_surface.tabulate_dof_coordinates()
    values=[usurf(point) for point in coordext]
    values=np.array(values)

    return values

def solve(nn, radius,path,seed3d,seed1d):
    
    min=2000
    max=12000

    meshV, meshQ, markers,ds1d = get_mesh(nn, radius, path)
    u_3d=boundary_cond3d(min, max, meshV, seed3d)
    facce=val_facce(min,max,meshV,seed3d)
    u_1d=boundary_cond1d(min,max,meshQ,meshV,seed1d)
    usurf= u_surf(u_3d, meshV)

    (AD, M), b, W, C, V_DOF, AAA = get_system(meshV, meshQ, u_3d, u_1d, ds1d,coupling_radius=radius)


    b_PET = ii_convert(b)
    shape = b_PET.size()  
    A_PET = ii_convert(AAA).mat()
    b_PET = b_PET.vec()
    u     = np.zeros((shape))
    tol   = 1e-15
    u_PET = PETSc.Vec().createWithArray(u)

    solver = PETSc.KSP().create()
    solver.setOperators(A_PET)
    solver.setType(PETSc.KSP.Type.GMRES)  # <--Choose the solver type
    solver.setFromOptions()  # <--Allow setting options from the command line or a file
    solver.setTolerances(rtol=tol)
    solver.setPCSide(1)
    solver.view()

    # Set preconditioner
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")  # Set to use AMG

    solver.solve(b_PET, u_PET)
    
    # print
    print ('iterations = ',               solver.getIterationNumber())
    print ('residual = ', '{:.2e}'.format(solver.getResidualNorm()))#  %.2E
    print ('converge reason = ',          solver.getConvergedReason())
    print ('residuals at each iter = ',   solver.getConvergenceHistory())
    print ('precond type', pc.getType())
    # Print solver information
    num_iter = solver.getIterationNumber()
    print(f"Number of iterations: {num_iter}")

    print("\n------------------ System setup and assembly time: ", "\n")

    u_npy=np.array(u_PET.getArray())
    n1=len(W[0].tabulate_dof_coordinates())
    n2=len(W[1].tabulate_dof_coordinates())
    
    u1_npy=u_npy[0:n1]
    u2_npy=u_npy[n1:n1+n2]

    coord1d=W[1].tabulate_dof_coordinates()
    u_dict={}
    for i in range(n2):
        key=tuple(coord1d[i])
        u_dict[key]=u2_npy[i]

    distanza=getGraphDist(meshQ, meshV,u_dict)   
    
    u1=Function(W[0])
    u1.vector()[:]=u1_npy
    u2=Function(W[1])
    u2.vector()[:]=u2_npy


    File('net_sol3d_def3.pvd') << u1
    File('net_sol1d_def3.pvd') << u2
    File('u_3d_def3.pvd') << u_3d
    File('u_1d_def3.pvd') <<u_1d

    file = XDMFFile(MPI.comm_world, f"dist_markers.xdmf")
    file.parameters["flush_output"] = True
    # Write the mesh and mesh function to the XDMF file
    file.write(distanza)
    # Close the XDMFFile
    file.close()

    return distanza, u1_npy, usurf,facce

if __name__ == '__main__':
    
    import numpy as np
    from petsc4py import PETSc

    radius = 1e-2
    #mesh parameters-----------------------------------------------------
    nn = 20
    
    dist_array=[]
    sol_3d=[]
    boundcond=[]
    sixtuple=[]

    for t in range(1):
        count=t+7
        path_to_1Dmesh = f'net_diedri/{count}_'
        for i in range (1):
            seed3d=i+5*count
            for k in range(6):
                seed1d=k*count+3
                dist, u3d, usurf,facce=solve(nn,radius,path_to_1Dmesh,seed3d,seed1d)
                distanza=dist.array()
                #distanze.append(dist)
                dist_array.append(distanza)
                sol_3d.append(u3d)
                boundcond.append(usurf)
                sixtuple.append(facce)
                print('-------------------------done------------------------')

    #distanze=np.array(distanze)
    dist_array=np.array(dist_array)
    sol_3d=np.array(sol_3d)
    boundcond=np.array(boundcond)
    sixtuple=np.array(sixtuple)

    #np.save("distanze.npy",distanze)
    np.save("10distarray157.npy", dist_array)
    np.save("10sol3d157.npy",sol_3d)
    np.save("10boundcond157.npy", boundcond) 
    np.save("10sixtuple157.npy", sixtuple)       

    
