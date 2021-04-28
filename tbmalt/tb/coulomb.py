"""Deal with coulombic interactions."""
import torch
import numpy as np
from tbmalt.common.structures.periodic import Periodic
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_bohr = 0.529177249


class Coulomb:
    """Calculate the coulombic interaction in periodic geometry.

    Arguments:
        geometry: Object for calculation, storing the data of input geometry.
        latvec: Lattice vector describing the geometry of periodic geometry.
        recvec: Reciprocal lattice vectors.

    Keyword Args:
        tol_ewald: EWald tolerance.
        nsearchiter: Maximum of iteration for searching alpha, maxg and maxr.

    Return:
        invrmat: 1/R matrix for the periodic geometry.
        alpha: Optimal alpha for the Ewald summation
        maxr: The longest real space vector that gives a bigger
              contribution to the EWald sum than tolerance.
        maxg: The longest reciprocal vector that gives a bigger
              contribution to the EWald sum than tolerance.

    """

    def __init__(self, geometry: object, periodic: object, **kwargs):
        self.geometry = geometry
        self.latvec = geometry.cell
        self.mask_pe = periodic.mask_pe
        self.natom = self.geometry.size_system
        self.coord = self.geometry.positions
        self.cellvol = periodic.cellvol
        self.recvec = periodic.recvec
        self.tol_ewald = kwargs.get('tol_ewald', 1e-9)
        self.nsearchiter = kwargs.get('nsearchiter', 30)

        # Optimal alpha for the Ewald summation
        self.alpha = self.get_alpha()

        # The longest real space vector
        self.maxr = self.get_maxr()

        # The longest reciprocal vector
        self.maxg = self.get_maxg()

        # The updated lattice points
        self.rcellvec_ud, self.ncell_ud = self.update_latvec()

        # The updated neighbour lists
        self.distmat, self.neighbour = self.update_neighbour()

        # Real part of the Ewald summation
        self.ewald_r, self.mask_g = self.invr_periodic_real()

        # Reciprocal part of the Ewald summation
        self.ewald_g = self.invr_periodic_reciprocal()

        # 1/R matrix for the periodic geometry
        self.invrmat = self.invr_periodic()
        if not self.mask_pe.all():
            self.invrmat[~self.mask_pe] = 0

    def update_latvec(self):
        """Update the lattice points for reciprocal Ewald summation."""
        update = Periodic(self.geometry, self.recvec, cutoff=self.maxg,
                          distance_extention=0, positive_extention=0, negative_extention=0, unit='Bohr')
        return update.rcellvec, update.ncell

    def update_neighbour(self):
        """Update the neighbour lists for real Ewald summation."""
        update = Periodic(self.geometry, self.latvec, cutoff=self.maxr,
                          distance_extention=0, unit='Bohr')
        return update.periodic_distances, update.neighbour

    def add_energy(self, shiftperatom, deltaq_atom, escc):
        """Add contribution from coulombic interaction to scc energy."""
        return escc + 0.5 * shiftperatom * deltaq_atom

    def invr_periodic(self):
        """Calculate the 1/R matrix for the periodic geometry."""
        # Extra contribution for self interaction
        extra = torch.stack([torch.eye(self.ewald_r.size(1)) * 2.0 * self.alpha[ii] / np.sqrt(np.pi)
                             for ii in range(self.ewald_r.size(0))])
        invr = self.ewald_r + self.ewald_g - extra
        invr[self.mask_g] = 0
        return invr

    def invr_periodic_reciprocal(self):
        """Calculate the reciprocal part of 1/R matrix for the periodic geometry."""
        # Lattice points for the reciprocal sum.
        n_low = torch.ceil(torch.clone(self.ncell_ud / 2.0))

        # Large values are padded in the end of short vectors.
        gvec_tem = pack([torch.unsqueeze(self.rcellvec_ud[
            ibatch, int(n_low[ibatch]):int(2 * n_low[ibatch] - 1)], 0)
            for ibatch in range(self.rcellvec_ud.size(0))], value=1e10)

        dd2 = torch.sum(torch.clone(gvec_tem) ** 2, -1)
        mask = torch.cat([torch.unsqueeze(dd2[ibatch] < self.maxg[ibatch] ** 2, 0)
                          for ibatch in range(self.rcellvec_ud.size(0))])
        gvec = pack([gvec_tem[ibatch, mask[ibatch]]
                     for ibatch in range(self.rcellvec_ud.size(0))], value=1e10)

        # Vectors for calculating the reciprocal Ewald sum
        coord1 = pack([self.coord[ibatch].repeat(self.coord.size(1), 1)
                      for ibatch in range(self.coord.size(0))])
        coord2 = pack([torch.cat(([self.coord[ibatch, iatom].repeat(self.coord.size(1), 1)
                      for iatom in range(self.natom[ibatch])]))
                      for ibatch in range(self.coord.size(0))])
        rr = coord1 - coord2

        # The reciprocal Ewald sum
        recsum = self.ewald_reciprocal(rr, gvec, self.alpha, self.cellvol)
        ewald_g = pack([torch.unsqueeze(
            recsum[ibatch] - np.pi / (self.cellvol[ibatch] * self.alpha[ibatch] ** 2), 0)
            for ibatch in range(self.alpha.size(0))])
        ewald_g = torch.reshape(ewald_g, (
            self.alpha.size(0), self.coord.size(1), self.coord.size(1)))
        ewald_g[self.mask_g] = 0
        return ewald_g

    def invr_periodic_real(self):
        """Calculate the real part of 1/R matrix for the periodic geometry."""
        ewaldr_tmp = self.ewald_real()

        # Mask for summation
        mask = ewaldr_tmp < float('inf')
        mask_real = self.neighbour & mask
        ewaldr_tmp[~mask_real] = 0
        ewald_r = torch.sum(ewaldr_tmp, 1)

        # Mask used for calculation of reciprocal part
        mask_g = ewald_r == 0
        return ewald_r, mask_g

    def get_alpha(self):
        """Get optimal alpha for the Ewald sum."""
        # Ewald parameter
        alphainit = torch.tensor([1.0e-8]).repeat(self.latvec.shape[0])
        alpha = torch.clone(alphainit)

        # Length of the shortest vector in reciprocal space
        min_g = torch.sqrt(torch.min(torch.sum(self.recvec ** 2, -1), 1).values)

        # Length of the shortest vector in real space
        min_r = torch.sqrt(torch.min(torch.sum(self.latvec ** 2, -1), 1).values)

        # Difference between reciprocal and real parts of the decrease of Ewald sum
        diff = self.diff_rec_real(alpha, min_g, min_r, self.cellvol)
        ierror = 0

        # Mask for batch calculation
        mask = diff < - self.tol_ewald

        # Loop to find the alpha
        while ((alpha[mask] < float('inf')).all()):
            alpha[mask] = 2.0 * alpha[mask]
            diff[mask] = self.diff_rec_real(alpha[mask], min_g[mask],
                                            min_r[mask], self.cellvol[mask])
            mask = diff < - self.tol_ewald
            if (~mask).all() == True:
                break
        if torch.max(alpha >= float('inf')):
            ierror = 1
        elif torch.max(alpha == alphainit):
            ierror = 2

        if ierror == 0:
            alphaleft = 0.5 * alpha
            mask = diff < self.tol_ewald
            while((alpha[mask] < float('inf')).all()):
                alpha[mask] = 2.0 * alpha[mask]
                diff[mask] = self.diff_rec_real(alpha[mask], min_g[mask],
                                                min_r[mask], self.cellvol[mask])
                mask = diff < self.tol_ewald
                if (~mask).all() == True:
                    break

        if torch.max(alpha >= float('inf')):
            ierror = 3

        if ierror == 0:
            alpharight = alpha
            alpha = (alphaleft + alpharight) / 2.0
            iiter = 0
            diff = self.diff_rec_real(alpha, min_g, min_r, self.cellvol)
            mask = torch.abs(diff) > self.tol_ewald
            while(iiter <= self.nsearchiter):
                mask_neg = diff < 0
                alphaleft[mask_neg] = alpha[mask_neg]
                alpharight[~mask_neg] = alpha[~mask_neg]
                alpha[mask] = (alphaleft[mask] + alpharight[mask]) / 2.0
                diff[mask] = self.diff_rec_real(alpha[mask], min_g[mask],
                                                min_r[mask], self.cellvol[mask])
                mask = torch.abs(diff) > self.tol_ewald
                iiter += 1
                if (~mask).all() == True:
                    break
            if iiter > self.nsearchiter:
                ierror = 4

        if ierror != 0:
            raise ValueError('Fail to get optimal alpha for Ewald sum.')
        return alpha

    def get_maxg(self):
        """Get the longest reciprocal vector that gives a bigger
           contribution to the EWald sum than tolerance."""
        ginit = torch.tensor([1.0e-8]).repeat(self.alpha.shape[0])
        ierror = 0
        xx = torch.clone(ginit)
        yy = self.gterm(xx, self.alpha, self.cellvol)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while ((xx[mask] < float('inf')).all()):
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self.gterm(xx[mask], self.alpha[mask], self.cellvol[mask])
            mask = yy > self.tol_ewald
            if (~mask).all() == True:
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == ginit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self.gterm(xleft, self.alpha, self.cellvol)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while(iiter <= self.nsearchiter):
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self.gterm(xx[mask], self.alpha[mask], self.cellvol[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all() == True:
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')
        return xx

    def get_maxr(self):
        """Get the longest real space vector that gives a bigger
           contribution to the EWald sum than tolerance."""
        rinit = torch.tensor([1.0e-8]).repeat(self.alpha.shape[0])
        ierror = 0
        xx = torch.clone(rinit)
        yy = self.rterm(xx, self.alpha)

        # Mask for batch
        mask = yy > self.tol_ewald

        # Loop
        while ((xx[mask] < float('inf')).all()):
            xx[mask] = xx[mask] * 2.0
            yy[mask] = self.rterm(xx[mask], self.alpha[mask])
            mask = yy > self.tol_ewald
            if (~mask).all() == True:
                break
        if torch.max(xx >= float('inf')):
            ierror = 1
        elif torch.max(xx == rinit):
            ierror = 2

        if ierror == 0:
            xleft = xx * 0.5
            xright = torch.clone(xx)
            yleft = self.rterm(xleft, self.alpha)
            yright = torch.clone(yy)
            iiter = 0
            mask = (yleft - yright) > self.tol_ewald

            while(iiter <= self.nsearchiter):
                xx[mask] = 0.5 * (xleft[mask] + xright[mask])
                yy[mask] = self.rterm(xx[mask], self.alpha[mask])
                mask_yy = yy >= self.tol_ewald
                xleft[mask_yy] = xx[mask_yy]
                yleft[mask_yy] = yy[mask_yy]
                xright[~mask_yy] = xx[~mask_yy]
                yright[~mask_yy] = yy[~mask_yy]
                mask = (yleft - yright) > self.tol_ewald
                iiter += 1
                if (~mask).all() == True:
                    break
            if iiter > self.nsearchiter:
                ierror = 3
        if ierror != 0:
            raise ValueError('Fail to get maxg for Ewald sum.')
        return xx

    def ewald_reciprocal(self, rr, gvec, alpha, vol):
        """Calculate the reciprocal part of the Ewald sum."""
        g2 = torch.sum(gvec ** 2, -1)
        dot = torch.tensor([[[igvec[0] @ irr[0] for igvec in zip(gvec[ibatch])]
                             for irr in zip(rr[ibatch])] for ibatch in range(alpha.size(0))])
        recsum = torch.cat([torch.unsqueeze(torch.sum((torch.exp(
            - g2[ibatch] / (4.0 * alpha[ibatch] ** 2)) / g2[ibatch]) *
            torch.cos(dot[ibatch]), -1), 0) for ibatch in range(alpha.size(0))])
        return torch.cat([torch.unsqueeze(2.0 * recsum[ibatch] * 4.0 * np.pi / vol[ibatch], 0)
                         for ibatch in range(alpha.size(0))])

    def ewald_real(self):
        """Batch calculation of the Ewald sum in the real part for a certain vector length."""
        return pack([torch.erfc(self.alpha[ibatch] * self.distmat[ibatch]) / self.distmat[ibatch]
                     for ibatch in range(self.alpha.size(0))])

    def diff_rec_real(self, alpha, min_g, min_r, cellvol):
        """Returns the difference between reciprocal and real parts of the decrease of Ewald sum."""
        return (self.gterm(4.0 * min_g, alpha, cellvol) - self.gterm(
            5.0 * min_g, alpha, cellvol)) - (self.rterm(2.0 * min_r, alpha) -
                                             self.rterm(3.0 * min_r, alpha))

    def gterm(self, len_g, alpha, cellvol):
        """Returns the maximum value of the Ewald sum in the
           reciprocal part for a certain vector length."""
        return 4.0 * np.pi * (torch.exp((-0.25 * len_g ** 2) / (alpha ** 2))
                              / (cellvol * len_g ** 2))

    def rterm(self, len_r, alpha):
        """Returns the maximum value of the Ewald sum in the
           real part for a certain vector length."""
        return torch.erfc(alpha * len_r) / len_r
