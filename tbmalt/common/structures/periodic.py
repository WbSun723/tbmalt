"""Deal with periodic conditions."""
import torch
import numpy as np
Tensor = torch.Tensor
_bohr = 0.529177249
distFudge = 1.0

def get_cell_translations_3d(self, latvec, kpoint, hs_cutoff, posExt=1, negExt=1):
    """Calculate the translation vectors for cells for 3D periodic boundary condition"""
   
    #Lattice vectors
    latvec = latvec / _bohr
    
    #Global cutoff for the diatomic interactions
    cutoff = hs_cutoff + distFudge
    
    #Unit cell volume
    self.dataset['cellvol'] = abs(torch.det(latvec))
    
    #Inverse lattice vectors (reciprocal lattice vectors in units of 2 pi, column)
    self.dataset['invlatvec'] = torch.inverse(latvec)
    
    #Reciprocal lattice vectors in units of 2 pi
    self.dataset['recvec2p'] = torch.transpose(self.dataset['invlatvec'], 0, 1)
    
    #Reciprocal lattice vectors
    self.dataset['recvec'] = 2.0 * np.pi * self.dataset['recvec2p']
    
    #Reciprocal lattice unit cell volume
    self.dataset['reccellvol'] = abs(torch.det(self.dataset['recvec']))
    
    #Get ranges of periodic boundary condition
    ranges = torch.zeros((2,3))
    iTmp = torch.stack([torch.floor(cutoff *\
                        torch.sqrt(sum(self.dataset['invlatvec'][:,ii]**2))) for ii in range (3)])
    ranges[0] = torch.stack([-(negExt + iTmp[ii]) for ii in range(3)])
    ranges[1] = torch.stack([(posExt + iTmp[ii]) for ii in range(3)])
    
    #Number of lattice cells
    self.dataset['ncell'] = (2.0*iTmp[0]+posExt+negExt+1) * (2.0*iTmp[1]+posExt+negExt+1) *\
                                                            (2.0*iTmp[1]+posExt+negExt+1)
    
    #Cell translation vectors in relative coordinates
    cellvec = torch.zeros((int(self.dataset['ncell']), 3))
    n = 0
    for ii in range(int(ranges[0,0]), int(ranges[1,0]+1)):
        for jj in range(int(ranges[0,1]), int(ranges[1,1]+1)):
            for kk in range(int(ranges[0,2]), int(ranges[1,2]+1)):
                cellvec[n][0] = ii
                cellvec[n][1] = jj
                cellvec[n][2] = kk
                n += 1
    self.dataset['cellvec'] = cellvec
    
    #Cell translation vectors in absolute units
    self.dataset['rcellvec'] = torch.stack([torch.matmul(torch.transpose(latvec,0,1), cellvec[ii])\
                                     for ii in range(int(self.dataset['ncell']))])
    
    
    
    
    
    
    
    pass

def get_neighbour(distance: Tensor, cutoff):
    """"""
    pass
