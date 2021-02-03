"""Interface to ASE."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import numpy as np
import torch as t
from ase import Atoms
import subprocess
from ase.calculators.aims import Aims
DFTB_ENERGY = {"H": -0.238600544, "C": -1.398493891, "N": -2.0621839400,
               "O": -3.0861916005}
AIMS_ENERGY = {"H": -0.45891649, "C": -37.77330663, "N": -54.46973501,
               "O": -75.03140052}
core_charge = {"H": 0, "C": 2, "N": 2, "O": 2}
HIRSH_VOL = {"H": 10.31539447, "C": 38.37861207, "N": 29.90025370,
             "O": 23.60491416}


class AseAims:
    """RunASEAims will run FHI-aims with both batch or single calculations.

    Arguments:
        path_to_aims: Joint path and executable binary FHI-aims.
        aims_specie: path to species_defaults.
        """

    def __init__(self, path_to_aims, aims_specie):
        self.aims = path_to_aims
        self.aims_specie = aims_specie

        # check if aims exists
        if not os.path.isfile(self.aims):
            raise FileNotFoundError('%s not found' % self.aims)

        self.set_env()  # set environment before calculations

    def set_env(self):
        """Set the environment before DFTB calculations with ase."""
        # copy binary to current path
        os.system('cp ' + self.aims + ' ./aims.x')

        # get the current binary path and name
        path_bin = os.path.join(os.getcwd(), 'aims.x')
        self.aimsout = os.path.join(os.getcwd(), 'aims.out')

        # set ase environemt
        os.environ['ASE_AIMS_COMMAND'] = path_bin + ' > PREFIX.out'
        os.environ['AIMS_SPECIES_DIR'] = self.aims_specie

    def run_aims(self, positions, symbols, properties):
        """Run batch systems with ASE-DFTB."""
        self.symbols = symbols
        results = {}
        for iproperty in properties:
            results[iproperty] = []

        for iposition, isymbol in zip(positions, symbols):
            print('ibatch', 'self.symbols[ibatch]', isymbol)

            # run each molecule in batches
            self.ase_iaims(iposition, isymbol)
            self.nat = len(iposition)

            # process each result (overmat, eigenvalue, eigenvect, dipole)
            for iproperty in properties:
                func = getattr(AseAims, iproperty)
                results[iproperty].append(func(self, isymbol))

        # remove DFTB files        self.remove()
        return results

    def ase_iaims(self, position, symbol):
        """Build Aims input by ASE."""
        # set Atoms with molecule specie and coordinates
        # print("moleculespecie", moleculespecie, 'coor', coor)
        mol = Atoms(symbol, positions=position)

        cal = Aims(xc='PBE',
                   output=['dipole', 'mulliken'],
                   sc_accuracy_etot=1e-6,
                   sc_accuracy_eev=1e-3,
                   sc_accuracy_rho=1e-6,
                   sc_accuracy_forces=1e-4,
                   many_body_dispersion=' ',
                   command="mpirun aims.x > aims.out")

        # get calculators
        mol.calc = cal
        try:
            mol.get_potential_energy()
        except UnboundLocalError:
            mol.calc.__dict__["parameters"]['Options_WriteHS'] = 'No'
            mol.get_potential_energy()

    def homo_lumo(self, symbol):
        """read homo and lumo"""
        commh = "grep 'Highest occupied state (VBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        self.homo = subprocess.check_output(commh, shell=True).decode('utf-8')
        comml = "grep 'Lowest unoccupied state (CBM) at' " + \
            self.aimsout + " | tail -n 1 | awk '{print $6}'"
        self.lumo = subprocess.check_output(comml, shell=True).decode('utf-8')
        return np.asarray([float(self.homo), float(self.lumo)])

    def dipole(self, symbol):
        """Read dipole."""
        commdip = "grep 'Total dipole moment' "
        cdx = commdip + self.aimsout + " | awk '{print $7}'"
        cdy = commdip + self.aimsout + " | awk '{print $8}'"
        cdz = commdip + self.aimsout + " | awk '{print $9}'"
        idipx = float(subprocess.check_output(cdx, shell=True).decode('utf-8'))
        idipy = float(subprocess.check_output(cdy, shell=True).decode('utf-8'))
        idipz = float(subprocess.check_output(cdz, shell=True).decode('utf-8'))
        return np.asarray([idipx, idipy, idipz])

    def total_energy(self, symbol):
        """get total energy, formation energy."""
        comme = "grep 'Total energy                  :' " + self.aimsout + \
            " | tail -n 1 | awk '{print $5}'"
        self.E_tot = float(
            subprocess.check_output(comme, shell=True).decode('utf-8'))
        return self.cal_optfor_energy(self.E_tot, symbol)

    def alpha_mbd(self, symbol):
        """Read polarizability."""
        commp = "grep -A " + str(self.nat + 1) + \
            " 'C6 coefficients and polarizabilities' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $6}'"
        ipol = subprocess.check_output(commp, shell=True).decode('utf-8')
        return np.asarray([float(i) for i in ipol.split('\n')[:-1]])

    def charge(self, symbol):
        """Read charges."""
        commc = "grep -A " + str(self.nat) + \
            " 'atom       electrons          charge' " + self.aimsout + \
                " | tail -n " + str(self.nat) + " | awk '{print $3}'"
        icharge = subprocess.check_output(commc, shell=True).decode('utf-8')
        charge = np.asarray([float(i) for i in icharge.split('\n')[:-1]])
        return self.remove_core_charge(charge, symbol)

    def hirshfeld_volume(self, symbol):
        """Read Hirshfeld volume."""
        cvol = "grep 'Hirshfeld volume        :' " + self.aimsout + \
            " | awk '{print $5}'"
        ivol = subprocess.check_output(cvol, shell=True).decode('utf-8')
        return np.asarray([float(i) for i in ivol.split('\n')[:-1]])

    def hirshfeld_volume_ratio(self, symbol):
        """Read Hirshfeld volume ratio."""
        hirshfeld_volume = self.hirshfeld_volume(symbol)
        return np.array([hirshfeld_volume[ia] / HIRSH_VOL[symbol[ia]]
                         for ia in range(self.nat)])

    def remove(self):
        """Remove all DFTB data after calculations."""
        os.system('rm aims.out control.in geometry.in Mulliken.out parameters.ase')

    def cal_optfor_energy(self, energy, symbol):
        """Calculate formation energy."""
        return energy - sum([AIMS_ENERGY[symbol[ia]] for ia in range(self.nat)])

    def remove_core_charge(self, charge, symbol):
        """Calculate formation energy."""
        return np.array([charge[ia] - core_charge[symbol[ia]]
                         for ia in range(self.nat)])


def get_matrix(filename):
    """Read DFTB+ hamsqr1.dat and oversqr.dat."""
    text = ''.join(open(filename, 'r').readlines())
    string = re.search('(?<=MATRIX\n).+(?=\n)', text, flags = re.DOTALL).group(0)
    out = np.array([[float(i) for i in row.split()] for row in string.split('\n')])
    return t.from_numpy(out)


def get_eigenvec(filename):
    """Read DFTB+ eigenvec.out."""
    string = []
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string_ = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    del string_[1::2]
    [string.append(float(ii)) for ii in string_]
    nstr = int(np.sqrt(len(string)))

    # transfer list to ==> numpy(float64) ==> torch
    eigenvec = np.asarray(string).reshape(nstr, nstr)
    return t.from_numpy(eigenvec).T


def get_eigenvalue(filename):
    """Read DFTB+ band.out."""
    text = ''.join(open(filename, 'r').readlines())

    # only read float
    string = re.findall(r"[-+]?\d*\.\d+", text)

    # delete even column
    eigenval_ = [float(ii) for ii in string[1::2]]
    occ_ = [float(ii) for ii in string[0::2]][1:]  # remove the first value

    # transfer list to ==> numpy(float64) ==> torch
    eigenval = t.from_numpy(np.asarray(eigenval_))
    occ = t.from_numpy(np.asarray(occ_))
    humolumo = np.asarray([eigenval[np.where(occ != 0)[0]][-1],
                           eigenval[np.where(occ == 0)[0]][0]])
    return eigenval, occ, humolumo


def read_detailed_out(natom):
    """Read DFTB+ output file detailed.out."""
    qatom, dip = [], []
    text = ''.join(open('detailed.out', 'r').readlines())
    E_tot_ = re.search('(?<=Total energy:).+(?=\n)',
                       text, flags=re.DOTALL | re.MULTILINE).group(0)
    E_tot = re.findall(r"[-+]?\d*\.\d+", E_tot_)[0]

    # read charge
    text2 = re.search('(?<=Atom       Population\n).+(?=\n)',
                      text, flags=re.DOTALL | re.MULTILINE).group(0)
    qatom_ = re.findall(r"[-+]?\d*\.\d+", text2)[:natom]
    [qatom.append(float(ii)) for ii in qatom_]

    # read dipole (Debye)
    text3 = re.search('(?<=Dipole moment:).+(?=\n)',
                      text, flags=re.DOTALL | re.MULTILINE).group(0)
    # if tail is [-3::], read Debye dipole, [:3] will read au dipole
    dip_ = re.findall(r"[-+]?\d*\.\d+", text3)[:3]
    [dip.append(float(ii)) for ii in dip_]

    return float(E_tot), \
        t.from_numpy(np.asarray(qatom)), t.from_numpy(np.asarray(dip))
