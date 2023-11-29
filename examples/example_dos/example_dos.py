import os
from os.path import exists
from typing import Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import h5py
import matplotlib.pyplot as plt

from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.io.dataset import DataSetIM
from tbmalt.physics.dftb.properties import dos
import tbmalt.common.maths as tb_math
from tbmalt.common.batch import pack
from tbmalt.data.units import energy_units


Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# ================ #
# STEP 1: Settings #
# ================ #

# 1.1: System settings
# --------------------

# Provide a list of systems upon which TBMaLT is to be run
training_size = 1
testing_size = 1
targets = ['dos']

# Provide information about the orbitals on each atom; this is keyed by atomic
# numbers and valued by azimuthal quantum numbers like so:
#   {Z₁: [ℓᵢ, ℓⱼ, ..., ℓₙ], Z₂: [ℓᵢ, ℓⱼ, ..., ℓₙ], ...}
shell_dict = {14: [0, 1, 2]}

# Identify which species are present
species = torch.tensor([14])
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# 1.2: Model settings
# -------------------
# Location at which the DFTB parameter set database is located
parameter_db_path = './siband.hdf5'

# Should fitting be performed here?
fit_model = True

# Should test the trained mode?
test = True

# Number of fitting cycles, number of batch size each cycle
number_of_epochs = 500
n_batch = 1

# learning rate
lr = 0.000005

# type of loss function
loss_function = 'Hellinger'

# Location of a file storing the properties that will be fit to.
target_path = './dataset_dos.h5'

# Choose which training and testing dataset to be loaded
# To run training and testing using the rattled type silicon systems, target_run
#    can be set as 'run1', 'run2' and 'run3'. To run the transferability test,
#    target_run should use 'run_transfer'.
target_run = 'run2'

# Energy window for dos sampling
points = torch.linspace(-3.3, 1.6, 491) if target_run != 'run_transfer' else\
    torch.linspace(-4.7, 1.6, 631)

# Load the Hamiltonian feed model
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation='spline', requires_grad=True)

# Load the overlap feed model
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation='spline', requires_grad=True)

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)

# 1.3: Construct the SCC-DFTB calculator object
# ---------------------------------------------
mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2,
              'generations': 3, 'tolerance': 1e-10}
kwargs = {}
kwargs['mix_params'] = mix_params
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, supress_SCF_error=True,
                        **kwargs)


# ======================== #
# STEP 2: Data preparation #
# ======================== #


# 2.1 load data set
def load_target_data(path: str, group1: str, group2: str, properties: List,
                     size: int) -> Any:
    """Load fitting target data.

    Arguments:
        path: path to a database in which the fitting data can be found.
        group 1: string to select the training run.
        group 2: string to select training or testing data.
        properties: list to select properties.
        size: size to load data.

    Returns:
        targets: returns an <OBJECT> storing the data to which the model is to
            be fitted.
    """
    # Data could be loaded from a json file or an hdf5 file; use your own
    # discretion here. A dictionary might be the best object in which to store
    # the target data.

    # To adjust the new dataloader
    with h5py.File(path, 'a') as f:
        attrs = f[group1][group2].attrs
        labels = attrs.get('labels', attrs.get('label', None))
        if labels is not None:
            attrs.__delitem__('label')

    return DataSetIM.load_data_batch(path, group1 + '/' + group2, properties,
                                     pbc=True)

# 2.2 prepare training and testing data
class SiliconDataset(Dataset):
    """Class to build training and testing dataset."""

    def __init__(self, numbers, positions, latvecs, homo_lumos, eigenvalues):
        self.numbers = numbers
        self.positions = positions
        self.latvecs = latvecs
        self.homo_lumos = homo_lumos
        self.eigenvalues = eigenvalues

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        number = self.numbers[idx]
        position = self.positions[idx]
        latvec = self.latvecs[idx]
        homo_lumo = self.homo_lumos[idx]
        eigenvalue = self.eigenvalues[idx]
        system = {"number": number, "position": position, "latvec": latvec,
                  "homo_lumo": homo_lumo, "eigenvalue": eigenvalue, "idx": idx}

        return system


def prepare_data(run):
    """Prepare training and testing data."""
    try:
        os.mkdir('./result')
    except FileExistsError:
        pass

    # Select systems for training and testing
    dataloder_train = load_target_data(target_path, run, 'train', ['homo_lumos', 'eigenvalues'], training_size)
    dataloder_test = load_target_data(target_path, run, 'test', ['homo_lumos', 'eigenvalues'], testing_size)
    indice = torch.arange(training_size + testing_size).tolist()

    data_train = dataloder_train[indice[: training_size]]
    data_test = dataloder_test[indice[: testing_size]]

    numbers_train, positions_train, cells_train = (data_train.geometry.atomic_numbers,
                                                   data_train.geometry.positions,
                                                   data_train.geometry.cells)

    numbers_test, positions_test, cells_test = (data_test.geometry.atomic_numbers,
                                                data_test.geometry.positions,
                                                data_test.geometry.cells)
    # Build datasets
    dataset_train = SiliconDataset(numbers_train, positions_train, cells_train,
                                   data_train.data['homo_lumos'],
                                   data_train.data['eigenvalues'])
    dataset_test = SiliconDataset(numbers_test, positions_test, cells_test,
                                  data_test.data['homo_lumos'],
                                  data_test.data['eigenvalues'])

    # Calculate reference dos and implement sampling
    ref_ev = data_train.data['eigenvalues']
    fermi_train = data_train.data['homo_lumos'].mean(dim=-1)
    energies_train = fermi_train.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(
            training_size, 0)
    dos_ref = dos((ref_ev), energies_train, 0.09)
    data_train_dos = torch.cat((energies_train.unsqueeze(-1),
                                dos_ref.unsqueeze(-1)), -1)

    # Plot training range of training data
    energies_plot = torch.linspace(-18, 5, 500).repeat(training_size, 1)
    dos_plot = dos((ref_ev), energies_plot, 0.09)
    dos_plot_mean = dos_plot.mean(dim=0)
    dos_plot_std = dos_plot.std(dim=0)
    plt.plot(energies_plot[0] - fermi_train.mean(dim=0), dos_plot_mean,
             linewidth=1.5, color='tab:blue')
    plt.fill_between(energies_plot[0] - fermi_train.mean(dim=0),
                     dos_plot_mean + dos_plot_std, dos_plot_mean - dos_plot_std,
                     alpha=0.4, facecolor='tab:blue')
    plt.fill_between(energies_train.mean(dim=0) - fermi_train.mean(dim=0),
                     -3, 60, alpha=0.2, facecolor='darkred')
    plt.xlim(-6, 4)
    plt.ylim(-2, 40)
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on')
    plt.rcParams["font.family"] = "arial"
    plt.xlabel('E - $\mathregular{E_f}$ [eV]', fontsize=15)
    plt.ylabel('DOS [states / eV]', fontsize=15)
    plt.title("Training range of DFT reference", fontsize=13)
    plt.savefig('training_range.pdf', dpi=500, bbox_inches='tight')
    plt.show()

    return dataset_train, dataset_test, data_train_dos


# split the data according to batch size and number of processers
def data_split(rank, world_size, dataset, batch_size, pin_memory=False,
               num_workers=0):
    """Prepare the data for distributed environment."""
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


# ================= #
# STEP 3: Execution #
# ================= #
class HellingerLoss(nn.Module):
    """Use the Hellinger distance as a loss function for training."""

    def __init__(self):
        super().__init__()

    def forward(self, p, q):
        return tb_math.hellinger(p, q).sum(0)


def loss_fn(results, ref_dos, ibatch):
    """Calculate loss during training."""
    loss = 0.
    # Get type of loss function.
    if loss_function == 'MSELoss':
        criterion = nn.MSELoss(reduction='mean')
    elif loss_function == 'Hellinger':
        criterion = HellingerLoss()

    # Calculate the loss
    ref = ref_dos[..., 1][ibatch] if ref_dos.ndim == 3 else ref_dos[..., 1]
    fermi_dftb = getattr(results, 'homo_lumo').mean(dim=-1) / energy_units['ev']
    energies_dftb = fermi_dftb.unsqueeze(-1) + points.unsqueeze(0).repeat_interleave(
        n_batch, 0)
    dos_dftb = dos((getattr(results, 'eigenvalue')) / energy_units['ev'],
                   energies_dftb, 0.09)
    loss = loss + criterion(dos_dftb, ref)

    return loss


def dftb_results(numbers, positions, cells, **kwargs):
    """Perform forward DFTB calculatoins."""
    # Whether do original DFTB calculations
    dftb = kwargs.get('dftb', False)

    # Build objects for DFTB calculations
    geometry = Geometry(numbers, positions, cells, units='a',
                        cutoff=torch.tensor([18.0]))
    orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict, shell_resolved=False)

    if not dftb:
        dftb_calculator(geometry, orbs)
    else:
        # Build new feeds
        h_feed_o = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                                        interpolation='spline')
        s_feed_o = SkFeed.from_database(parameter_db_path, species, 'overlap',
                                        interpolation='spline')
        mix_params = {'mix_param': 0.2, 'init_mix_param': 0.2,
              'generations': 3, 'tolerance': 1e-10}
        kwargs = {}
        kwargs['mix_params'] = mix_params
        dftb_calculator_o = Dftb2(h_feed_o, s_feed_o, o_feed, u_feed, **kwargs)
        dftb_calculator_o(geometry, orbs)

    return dftb_calculator if not dftb else dftb_calculator_o


def main(rank, world_size, train_dataset, data_train_dos):
    """ML training to optimize DFTB H and S matrix."""
    # Initial the model
    train_data = data_split(rank, world_size, train_dataset, batch_size=n_batch)
    h_var = [val.abcd for key, val in h_feed.off_sites.items()]
    s_var = [val.abcd for key, val in s_feed.off_sites.items()]
    variable = h_var + s_var
    optimizer = getattr(torch.optim, 'Adam')(variable, lr=lr)
    loss_list = []
    loss_list.append(0)

    # Training
    for epoch in range(number_of_epochs):
        _loss = 0
        print('epoch', epoch)
        train_data.sampler.set_epoch(epoch)
        for ibatch, data in enumerate(train_data):
            # Perform the forwards operation
            forward_cal = dftb_results(data['number'], data['position'],
                                       data['latvec'])
            loss = loss_fn(forward_cal, data_train_dos, data['idx'])
            _loss = _loss + loss
        optimizer.zero_grad()
        _loss.retain_grad()

        # Invoke the autograd engine
        _loss.backward(retain_graph=True)

        # Update the model
        optimizer.step()
        print("loss:", _loss)
        loss_list.append(_loss.detach())

    # Save the training loss
    f = open('./result/loss.dat', 'w')
    np.savetxt(f, torch.tensor(loss_list[1:]))
    f.close()


def test(rank, world_size, test_dataset):
    """Test the trained model."""
    try:
        os.mkdir('./result/test')
    except FileExistsError:
        pass

    test_data = data_split(rank, world_size, test_dataset, batch_size=1)
    energies_test = torch.linspace(-18.0, 5.0, 500)
    dos_pred_tot = []
    hl_pred_tot = []
    dos_dftb_tot = []
    hl_dftb_tot = []
    ref_hl = []
    ref_ev = []

    # Pred
    for ibatch, data in enumerate(test_data):
        scc_pred = dftb_results(data['number'], data['position'],
                                data['latvec'])
        ref_hl.append(data['homo_lumo'])
        ref_ev.append(data['eigenvalue'])
        fermi_pred = getattr(scc_pred, 'fermi_energy').detach() / energy_units['ev']
        hl_pred = getattr(scc_pred, 'homo_lumo').detach() / energy_units['ev']
        hl_pred_tot.append(hl_pred)
        eigval_pred = getattr(scc_pred, 'eigenvalue').detach() / energy_units['ev']
        dos_pred = dos((eigval_pred), energies_test, 0.09)
        f = open('./result/test/Pred_fermi' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, fermi_pred)
        f.close()
        f = open('./result/test/Pred_homo_lumo' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, hl_pred)
        f.close()
        pred_dos = torch.cat((energies_test.unsqueeze(-1),
                              dos_pred.unsqueeze(-1).squeeze(0)), -1)
        dos_pred_tot.append(dos_pred)
        f = open('./result/test/Pred_dos' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, pred_dos)
        f.close()
        f = open('./result/test/Pred_eigenvalue' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigval_pred.squeeze(0))
        f.close()

    # Pred mean
    dos_pred_mean = pack(dos_pred_tot).mean(dim=0)
    dos_pred_std = pack(dos_pred_tot).std(dim=0)
    f = open('./result/Pred_dos_mean.dat', 'w')
    np.savetxt(f, torch.cat((energies_test.unsqueeze(-1),
                             dos_pred_mean.unsqueeze(-1).squeeze(0)), -1))
    f.close()
    f = open('./result/Pred_dos_std.dat', 'w')
    np.savetxt(f, dos_pred_std.squeeze(0))
    f.close()
    f = open('./result/Pred_homo_lumo_mean.dat', 'w')
    np.savetxt(f, pack(hl_pred_tot).mean(dim=0).detach())
    f.close()
    f = open('./result/Pred_homo_lumo_std.dat', 'w')
    np.savetxt(f, pack(hl_pred_tot).std(dim=0).detach())
    f.close()

    # DFTB
    for ibatch, data in enumerate(test_data):
        scc_dftb = dftb_results(data['number'], data['position'],
                                data['latvec'], dftb=True)
        # dftb
        fermi_dftb = getattr(scc_dftb, 'fermi_energy').detach() / energy_units['ev']
        hl_dftb = getattr(scc_dftb, 'homo_lumo').detach() / energy_units['ev']
        hl_dftb_tot.append(hl_dftb)
        eigval_dftb = getattr(scc_dftb, 'eigenvalue').detach() / energy_units['ev']
        dos_dftb = dos((eigval_dftb), energies_test, 0.09)
        f = open('./result/test/dftb_fermi' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, fermi_dftb)
        f.close()
        f = open('./result/test/dftb_homo_lumo' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, hl_dftb)
        f.close()
        dftb_dos = torch.cat((energies_test.unsqueeze(-1),
                              dos_dftb.unsqueeze(-1).squeeze(0)), -1)
        dos_dftb_tot.append(dos_dftb)
        f = open('./result/test/dftb_dos' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, dftb_dos)
        f.close()
        f = open('./result/test/dftb_eigenvalue' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigval_dftb.squeeze(0))
        f.close()

    # DFTB mean
    dos_dftb_mean = pack(dos_dftb_tot).mean(dim=0)
    dos_dftb_std = pack(dos_dftb_tot).std(dim=0)
    f = open('./result/dftb_dos_mean.dat', 'w')
    np.savetxt(f, torch.cat((energies_test.unsqueeze(-1),
                             dos_dftb_mean.unsqueeze(-1).squeeze(0)), -1))
    f.close()
    f = open('./result/dftb_dos_std.dat', 'w')
    np.savetxt(f, dos_dftb_std.squeeze(0))
    f.close()
    f = open('./result/dftb_homo_lumo_mean.dat', 'w')
    np.savetxt(f, pack(hl_dftb_tot).mean(dim=0).detach())
    f.close()
    f = open('./result/dftb_homo_lumo_std.dat', 'w')
    np.savetxt(f, pack(hl_dftb_tot).std(dim=0).detach())
    f.close()

    # Plot results
    ref_hl_plot = pack(ref_hl)
    ref_ev_plot = pack(ref_ev)
    ref_dos_plot = dos((ref_ev_plot), energies_test, 0.09)
    ref_dos_mean = ref_dos_plot.mean(dim=0).squeeze(0)
    ref_dos_std = ref_dos_plot.std(dim=0).squeeze(0)
    ref_hl_mean = ref_hl_plot.mean(dim=0)
    ref_fermi_mean = ref_hl_mean.mean(dim=-1)

    # Original DFTB calculations
    dftb_fermi_mean = pack(hl_dftb_tot).mean(dim=0).detach().mean(dim=-1)
    dftb_dos_mean_plot = dos_dftb_mean.squeeze(0)
    dftb_dos_std_plot = dos_dftb_std.squeeze(0)
    plt.plot(energies_test - ref_fermi_mean, ref_dos_mean, label='DFT')
    plt.fill_between(energies_test - ref_fermi_mean, ref_dos_mean + ref_dos_std,
                     ref_dos_mean - ref_dos_std, alpha=0.5)
    plt.plot(energies_test - dftb_fermi_mean, dftb_dos_mean_plot,
             label='siband-1-1')
    plt.fill_between(energies_test - dftb_fermi_mean,
                     dftb_dos_mean_plot + dftb_dos_std_plot,
                     dftb_dos_mean_plot - dftb_dos_std_plot, alpha=0.5)
    plt.xlim(-6, 4)
    plt.ylim(-2, 40)
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on')
    plt.rcParams["font.family"] = "arial"
    plt.xlabel('E - $\mathregular{E_f}$ [eV]', fontsize=15)
    plt.ylabel('DOS [states / eV]', fontsize=15)
    plt.title("Before training", fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig('before_training.pdf', dpi=500, bbox_inches='tight')
    plt.show()

    # Prediction after training
    pred_fermi_mean = pack(hl_pred_tot).mean(dim=0).detach().mean(dim=-1)
    pred_dos_mean_plot = dos_pred_mean.squeeze(0)
    pred_dos_std_plot = dos_pred_std.squeeze(0)
    plt.plot(energies_test - ref_fermi_mean, ref_dos_mean, label='DFT')
    plt.fill_between(energies_test - ref_fermi_mean, ref_dos_mean + ref_dos_std,
                     ref_dos_mean - ref_dos_std, alpha=0.5)
    plt.plot(energies_test - pred_fermi_mean, pred_dos_mean_plot,
             label='spline')
    plt.fill_between(energies_test - pred_fermi_mean,
                     pred_dos_mean_plot + pred_dos_std_plot,
                     pred_dos_mean_plot - pred_dos_std_plot, alpha=0.5)
    plt.xlim(-6, 4)
    plt.ylim(-2, 40)
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on')
    plt.rcParams["font.family"] = "arial"
    plt.xlabel('E - $\mathregular{E_f}$ [eV]', fontsize=15)
    plt.ylabel('DOS [states / eV]', fontsize=15)
    plt.title("After training", fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig('after_training.pdf', dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dataset_train, dataset_test, data_train_dos = prepare_data(target_run)
    main(0, 1, dataset_train, data_train_dos)
    if test:
        test(0, 1, dataset_test)
