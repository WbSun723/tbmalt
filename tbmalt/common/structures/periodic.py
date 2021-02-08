"""Deal with periodic conditions."""
import torch
import numpy as np
Tensor = torch.Tensor
distfudge = 1.0

def get_cell_translations_3d(latvec: Tensor, sk_cutoff: float, posExt=1, negExt=1):
    """Calculate the translation vectors for cells for 3D periodic boundary condition.
    
    Arguments:
        latvec: Lattice vector describing the geometry of periodic system, with Bohr as unit.
        sk_cutoff: Interaction cutoff distance for reading  SK table.
        posExt: Extension for the positive lattice vectors.
        negEXt: Extension for the negative lattice vectors.
    
    Return:
        cutoff: Global cutoff for the diatomic interactions
        cellvol: Volume of the unit cell.
        reccellvol: Volume of the reciprocal lattice unit cell.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        ncell: Number of lattice cells.
    
    Examples:
        >>> from periodic import get_cell_translations_3d
        >>> import torch
        >>> _bohr = 0.529177249
        >>> latvec = torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]]) / _bohr
        >>> sk_cutoff = 499 * 0.02
        >>> cutoff, cellvol, reccellvol, cellvec, rcellvec, ncell = get_cell_translations_3d(latvec, sk_cutoff)
        >>> print(ncell)
        75
        >>> print(cellvec)
        tensor([[-1., -2., -2.],
                [-1., -2., -1.],
                [-1., -2.,  0.],
                [-1., -2.,  1.],
                [-1., -2.,  2.],
                      ...
                [ 1.,  2., -2.],
                [ 1.,  2., -1.],
                [ 1.,  2.,  0.],
                [ 1.,  2.,  1.],
                [ 1.,  2.,  2.]])
    
    """
    # Global cutoff for the diatomic interactions
    cutoff = sk_cutoff + distfudge
    
    # Unit cell volume
    cellvol = abs(torch.det(latvec))
    
    # Inverse lattice vectors (reciprocal lattice vectors in units of 2 pi, column)
    invlatvec = torch.inverse(latvec)
    
    # Reciprocal lattice vectors in units of 2 pi
    recvec2p = torch.transpose(invlatvec, 0, 1)
    
    # Reciprocal lattice vectors
    recvec = 2.0 * np.pi * recvec2p
    
    # Reciprocal lattice unit cell volume
    reccellvol = abs(torch.det(recvec))
    
    # Get ranges of periodic boundary condition
    ranges = torch.zeros((2, 3), dtype = torch.int)
    iTmp = torch.stack([torch.floor(cutoff *
                        torch.sqrt(sum(invlatvec[:, ii] ** 2))) for ii in range (3)])
    ranges[0] = torch.stack([-(negExt + iTmp[ii]) for ii in range(3)])
    ranges[1] = torch.stack([(posExt + iTmp[ii]) for ii in range(3)])
    
    # Length of the first, second and third column in ranges
    leng1, leng2, leng3 = ranges[1, :] - ranges[0, :] + 1
    
    # Number of lattice cells
    ncell = leng1 * leng2 * leng3
    
    # Cell translation vectors in relative coordinates
    cellvec = torch.zeros((ncell, 3))
    col3 = torch.linspace(ranges[0, 2], ranges[1, 2], leng3)
    col2 = torch.linspace(ranges[0, 1], ranges[1, 1], leng2)
    col1 = torch.linspace(ranges[0, 0], ranges[1, 0], leng1)
    cellvec[:, 2] = torch.cat(int(ncell / leng3) * [col3])
    col2 = col2.repeat(leng3, 1)
    col2 = torch.cat([(col2[:, ii]) for ii in range(leng2)])
    cellvec[:, 1] = torch.cat(int(ncell / (leng2 * leng3)) * [col2])
    col1 = col1.repeat(leng3 * leng2, 1)
    cellvec[:, 0] = torch.cat([(col1[:, ii]) for ii in range(leng1)])
    
    # Cell translation vectors in absolute units
    rcellvec = torch.stack([torch.matmul(torch.transpose(latvec, 0, 1), cellvec[ii])
                                     for ii in range(ncell)])
    
    # Number of lattice cells
    ncell = rcellvec.size(0)
    
    return cutoff, cellvol, reccellvol, cellvec, rcellvec, ncell

def get_neighbour(coord0: Tensor, species0: Tensor, rcellvec: Tensor, natom: int, ncell: int, cutoff: float):
    """Obtain neighbour list and species according to periodic boundary condition.
    
    Arguments:
        coord0: Coordinates of the atoms in central cell.
        species0: Species of the atoms in central cell.
        rcellvec: Cell translation vectors in absolute units.
        natom: NUmber of the atoms in central cell.
        ncell: Number of lattice cells
        cutoff: Global cutoff for the diatomic interactions.
    
    Return:
        neighdist2: Squared distance between atoms in central cell and other cells. 
        iposcentcell: Index of position for atoms in central cell.
        img2centcell: Mapping index of atoms in translated cells onto the atoms in central cell.
        icellvec: Index of cell vector for each atom.
        nallatom: Count of all atoms interacting with each atom in central cell.
        species: Species of all interacting atoms
    
    Notes: 
        neighdist2 is a 1D tensor. With three additional indices, i.e. iposcentcell, img2centcell 
        and icellvec, information of the interacting atoms for each element in neighdist2 can be 
        clearly defined.
    
    Examples:
        >>> from periodic import get_cell_translations_3d
        >>> from periodic import get_neighbour
        >>> import torch
        >>> _bohr = 0.529177249
        >>> latvec = torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]]) / _bohr
        >>> coord0 = torch.tensor([[0, 0, 0], [0, 4, 0]]) / _bohr
        >>> species0 = torch.tensor([[1], [1]])
        >>> sk_cutoff = 499 * 0.02
        >>> cutoff, cellvol, reccellvol, cellvec, rcellvec, ncell = get_cell_translations_3d(latvec, sk_cutoff)
        >>> neighdist2, iposcentcell, img2centcell, icellvec, nallatom, species = get_neighbour(
                                                          coord0, species0, rcellvec, natom, ncell, cutoff)
        >>> print(neighdist2)
        tensor([114.2741, 114.2741,  57.1370, 114.2741, 114.2741, 114.2741, 114.2741,
         57.1370, 114.2741, 114.2741,  57.1370, 114.2741, 114.2741,  57.1370,
        114.2741, 114.2741,  57.1370,   0.0000,  57.1370,  57.1370,   0.0000,
         57.1370, 114.2741, 114.2741,  57.1370, 114.2741, 114.2741,  57.1370,
        114.2741, 114.2741,  57.1370, 114.2741, 114.2741, 114.2741, 114.2741,
         57.1370, 114.2741, 114.2741])

    """
    # Square of the diatomic interaction cutoff
    cutoff2 = cutoff ** 2
    
    # Coordinates of atoms in all translated cells
    coord = torch.stack([(coord0 + rcellvec[ii]) for ii in range(ncell)])
    
    # Square of the distance between two atoms in translated and central cells respectively
    dist2 = torch.tensor([])
    
    # Mapping index of atoms in all translated cells onto the atoms in central cell
    imapall = torch.tensor([])
    imap = torch.linspace(0, natom-1, natom)
    mm = torch.stack([(torch.unsqueeze(imap, 1)) for ii in range(ncell)])
    
    # Index of cell vector for atoms in all translated cells
    icellall = torch.tensor([])
    icell = torch.stack([torch.cat(natom * [torch.tensor([ii])]) for ii in range(ncell)])
    cc = torch.unsqueeze(icell, -1)
    
    # Index of position for atoms in central cell
    iposiall = torch.tensor([])
    
    # Loop over all atoms in central cell
    for ii in range(natom):
        # Square of the distance between atoms in translated cells and atom ii in central cell
        dd = torch.sum((coord - coord0[ii]) ** 2, -1, keepdim=True)
        
        # Index of positon for atom ii in central cell
        pp = torch.unsqueeze(torch.cat(ncell * [torch.tensor([ii])]), -1)
        
        # dist2[ii, jj, kk] means square of the distance between atom jj in cell ii and atom kk in central cell
        # [ii, jj, kk] describes the same pair of atoms for imapall, icellall and iposiall
        dist2 = torch.cat((dist2, dd), dim=-1)
        imapall = torch.cat((imapall, mm), dim=-1)
        icellall = torch.cat((icellall, cc), dim=-1)
        iposiall = torch.cat((iposiall, torch.unsqueeze(pp, -1)), dim=-1)
    
    # Creating mask = True when dist2 <= cutoff2
    mask_dist2 = dist2.le(cutoff2)
    
    # Squared distance between atoms in central cell and other cells 
    # The position of atom in central is described by iposcentcell
    neighdist2 = torch.masked_select(dist2, mask_dist2)
    
    # Index of position for atoms in central cell
    iposcentcell = torch.masked_select(iposiall, mask_dist2).int()
    
    # Mapping index of atoms in translated cells onto the atoms in central cell
    img2centcell = torch.masked_select(imapall, mask_dist2).int()
    
    # Index of cell vector for each atom
    icellvec = torch.masked_select(icellall, mask_dist2).int()
    
    # Count of all atoms interacting with each atom in central cell
    nallatom = neighdist2.size(0)
    
    # Species of all interacting atoms
    species = torch.cat([(species0[img2centcell[ii]]) for ii in range(nallatom)])
    
    return neighdist2, iposcentcell, img2centcell, icellvec, nallatom, species