"""Constant data for general purpose."""
import torch

bohr = 0.529177249

atom_name = ["H", "He",
              "Li", "Be", "B", "C", "N", "O", "F", "Ne",
              "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
              "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
              "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
              "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
              "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
              "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
              "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os",
              "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At"]

val_elect = [1, 2,
             1, 2, 3, 4, 5, 6, 7, "Ne",
             1, 2, 3, 4, 5, 6, 7, "Ar",
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
             "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
             "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
             "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
             "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
             "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os",
             "Ir", "Pt", 11, "Hg", "Tl", "Pb", "Bi", "Po", "At"]

l_num = [0, 0,
          0, 0, 1, 1, 1, 1, 1, 1,
          0, 0, 1, 1, 1, 1, 1, 1,
          0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
          0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
          0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]

# Parametes for training
additional_para = {('Si_V', 'none'): ('./dataset/fhi-aims_si63v_hse_101.hdf',
                                      './dataset/fhi-aims_si63v_hse_101.hdf',
                                      './skf/skf_siband.hdf', True,
                                      torch.linspace(-3.0, 2.0, 501)),
                   ('Si_V', 'large'): ('./dataset/fhi-aims_si63v_relax_pbe.hdf',
                                       './dataset/fhi-aims_si512v_pbe.hdf',
                                       './skf/skf_siband.hdf', True,
                                       torch.linspace(-4.6, 6.9, 1151)),
                   ('Si_V', 'other defects'): ('./dataset/fhi-aims_si63v_hse_101.hdf',
                                               './dataset/fhi-aims_si65_interstitial_hse.hdf',
                                               './skf/skf_siband.hdf', True,
                                               torch.linspace(-3.0, 2.0, 501)),
                   ('SiC_V', 'none'): ('./dataset/fhi-aims_si32c31_hse_82.hdf',
                                       './dataset/fhi-aims_si32c31_hse_82.hdf',
                                       './skf/skf_pbc.hdf', False,
                                       torch.linspace(-4.1, 0.2, 431)),
                   ('SiC_V', 'large'): ('./dataset/fhi-aims_si32c31_relax_pbe.hdf',
                                        './dataset/fhi-aims_si256c255_pbe.hdf',
                                        './skf/skf_pbc.hdf', False,
                                        torch.linspace(-6.6, 0.2, 681)),
                   ('Si_I_100', 'large'): ('./dataset/fhi-aims_si65_100_relax_pbe.hdf',
                                           './dataset/fhi-aims_si513_100_pbe.hdf',
                                           './skf/skf_siband.hdf', True,
                                           torch.linspace(-2.6, 1.9, 451)),
                   ('Si_I_110', 'large'): ('./dataset/fhi-aims_si65_110_relax_pbe.hdf',
                                           './dataset/fhi-aims_si513_110_pbe.hdf',
                                           './skf/skf_siband.hdf', True,
                                           torch.linspace(-2.6, 1.9, 451))}
