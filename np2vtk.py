import numpy as np


class Slab(object):
    
    def __init__(self, xL, yL, zL, nx, ny, nz):
        self.xL = xL
        self.yL = yL
        self.zL = zL
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
    def vtk(self, filename, u, X):
        toVTK(filename, u, X, self.nx, self.ny, self.nz, self.xL, self.yL, self.zL)
        
    def coordinates(self):
        coords = []
        hx, hy, hz = self.xL/self.nx, self.yL/self.ny, self.zL/self.nz
        for k in range(self.nz+1):
            for j in range(self.ny+1):
                for i in range(self.nx+1):
                    coords.append([i*hx, j*hy, k*hz])
        return np.stack(coords)


def toVTK(filename, u, X, nx, ny, nz, xL = 1, yL = 1, zL = 0.15):
    """
    Numpy to VTK conversion of discrete functions.
    
    Input
         filename     (str)               Name of the file (WITHOUT extension).
         u           (numpy.ndarray)      1D array of length Nh, listing all function values
                                          at the Nh nodes (functional space dofs)
         X           (numpy.ndarray)      2D array of shape Nh x 3 listing all nodes coordinates
                                          (according to the ordering adopted for u)
         nx, ny, nz  (int)                Number of intervals per side
         xL, yL, zL  (float)              Sides length
         
    Output
        None. The VTK file (.vtu) will be stored locally.   
    """
    
    coordinates = []
    hx, hy, hz = xL/nx, yL/ny, zL/nz
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                coordinates.append([i*hx, j*hy, k*hz])

    coordinates = np.stack(coordinates)
    
    def findj(p, ref = coordinates):
        return np.argmin(np.linalg.norm(p-ref, axis = 1))

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = np.array([i*hx, j*hy, k*hz])
                A = [[findj(p), findj(p+[hx,0,0]), findj(p+[hx,hy,0]), findj(p+[hx,hy,hz])],
                     [findj(p), findj(p+[hx,0,0]), findj(p+[hx,0,hz]), findj(p+[hx,hy,hz])],
                     [findj(p), findj(p+[0,0,hz]), findj(p+[hx,0,hz]), findj(p+[hx,hy,hz])],
                     [findj(p), findj(p+[0,hy,0]), findj(p+[hx,hy,0]), findj(p+[hx,hy,hz])],
                     [findj(p), findj(p+[0,0,hz]), findj(p+[0,hy,hz]), findj(p+[hx,hy,hz])],
                     [findj(p), findj(p+[0,hy,0]), findj(p+[0,hy,hz]), findj(p+[hx,hy,hz])],
                    ]
                cells.append(A)
    cells = np.concatenate(cells)
    
    text = ('<?xml version="1.0"?>\n' +
        '<VTKFile type="UnstructuredGrid"  version="0.1"  >\n' +
        '<UnstructuredGrid>\n' + 
        ('<Piece  NumberOfPoints="%d" NumberOfCells="%d">\n' % (len(coordinates), len(cells))) + 
        '<Points>\n' + 
        '<DataArray  type="Float64"  NumberOfComponents="3"  format="ascii">')

    for c in coordinates:
        for v in c:
            text += str(v) + " "
        text += " "
        
        
    text += '</DataArray>\n</Points>\n<Cells>\n<DataArray  type="UInt32"  Name="connectivity"  format="ascii">'
    for c in cells:
        for v in c:
            text += str(v) + " "
        text += " "

    text += '</DataArray>\n<DataArray  type="UInt32"  Name="offsets"  format="ascii">'
    k = 0
    for c in cells:
        k += 4
        text += str(k) + " "
    text += '</DataArray>\n<DataArray  type="UInt8"  Name="types"  format="ascii">'

    for c in cells:
        text += "10 "
    text += '</DataArray>\n</Cells>\n<PointData  Scalars="Ct">\n<DataArray  type="Float64"  Name="Ct"  format="ascii">'

    J = [findj(x, ref = X) for x in coordinates]
    for j in J:
        text += (str(u[j]) + "  ")

    text += '</DataArray>\n</PointData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>'

    with open('%s.vtu' % filename, 'w') as f:
        print(text, file = f)