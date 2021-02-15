"""K points sampling for periodic system."""
import torch
import numpy as np
from tbmalt.common.batch import pack
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15)
Tensor = torch.Tensor
minlim = - 1e-4
maxlim = 1 - 1e-4

def build_supercell(coeffsandshifts: Tensor):
    """Build the super structure and genearte a set of special K-points.
    
    Arguments:
        coeffsandshifts: Parameters for building the super structure, consisting of coefficients and shifts.
    
    Return:
        Kpoints: Kpoints for periodic system.
        Kweights: The weights of each Kpoint.
        
    """
    # Coefficients used to build supercell vectors from original lattice vectors
    coeffs = coeffsandshifts[:, :3]

    # Shift of the grid along the three reciprocal lattice vectors
    shifts = coeffsandshifts[:, 3]
    
    # Number of Kpoints without reducing
    nkpoints = torch.det(coeffs)
    
    # Inverse coefficients
    invcoeffs = torch.transpose(torch.solve(torch.eye(
                coeffs.shape[-1]), coeffs)[0], -1, -2)
    
    # Get ranges for building supercell vectors
    imgrange1 = torch.min(torch.min(coeffs, dim=1).values, torch.tensor([0, 0, 0]))
    imgrange2 = torch.sum(coeffs, dim=1) - 1
    imgrange = torch.stack((imgrange1, imgrange2), 0)
    
    # rr: Relative coordinate with respect to the original reciprocal lattice
    leng = imgrange[1, :].long() - imgrange[0, :].long() + 1
    transvec = pack([torch.stack([
            torch.linspace(iran[0, 0], iran[1, 0],
                           ile[0]).repeat_interleave(ile[2] * ile[1]),
            torch.linspace(iran[0, 1], iran[1, 1],
                           ile[1]).repeat(ile[0]).repeat_interleave(ile[2]),
            torch.linspace(iran[0, 2], iran[1, 2],
                           ile[2]).repeat(ile[0] * ile[1])])
            for ile, iran in zip(leng, imgrange.transpose(1, 0))])
    rr = pack([(icf.transpose(0, 1) @ itv.T.unsqueeze(-1)).squeeze(-1)
                         for icf, itv in zip(invcoeffs, transvec)])
    
    # Mask for rr
    mask_rr = torch.all(torch.all(torch.all(rr.le(maxlim) * rr.ge(minlim), dim=-1), dim=1), dim=0)
    if mask_rr == False:
        raise ValueError('Failed to find all K-points.')
    
    # Add shift and obtain Kpoints
    ss = torch.stack([torch.matmul(icf, ishif) for icf, ishif in zip(invcoeffs, shifts)])
    kpoints = torch.stack([(irr + iaa) for irr, iaa in zip(rr, ss)])
    
    # Kweights
    iweight = 1 / nkpoints
    kweight = pack([torch.cat((int(nkpoints[ii]) * [torch.unsqueeze(torch.tensor([iweight[ii]]), dim=0)])) 
                    for ii in range(nkpoints.size(0))])
    
    return kpoints, kweight
    