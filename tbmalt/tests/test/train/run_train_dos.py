"""Example to run training."""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.common.parameter import Parameter
from tbmalt.common.data import additional_para
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.tb.sk import SKT
from tbmalt.tb.dftb.scc import Scc
from tbmalt.common.structures.periodic import Periodic
from tbmalt.tb.coulomb import Coulomb
from tbmalt.tb.properties import dos
from tbmalt.common.batch import pack
import tbmalt.common.maths as tb_math
import time

# To reproduce the results described in the paper, please use the following
#    combinations of ml_parameters.
# 1. General training and testing on Si_V systems:
#    params.ml_params['target'] = 'Si_V',
#    params.ml_params['transfer_type'] = 'none'
# 2. General training and testing on SiC_V systems:
#    params.ml_params['target'] = 'SiC_V',
#    params.ml_params['transfer_type'] = 'none'
# 3. Transferability testing on larger systems:
#    params.ml_params['target'] = 'Si_V', SiC_V', 'Si_I_100' or 'Si_I_110'
#    params.ml_params['transfer_type'] = 'large'
# 4. Transferability testing on predicting other types of defects (training on
#    vacancy and testing on interstitial):
#    params.ml_params['target'] = 'Si_V'
#    params.ml_params['transfer_type'] = 'other defects'

# Set general parameters for training and DFTB calculations
torch.set_printoptions(6, profile='full')
torch.set_default_dtype(torch.float64)
size_train, size_test = 1, 1
params = Parameter(ml_params=True)
params.ml_params['task'] = 'mlIntegral'
params.ml_params['steps'] = 2000
params.ml_params['lr'] = 0.000005
params.ml_params['test'] = True
# Taget systems include 'Si_V', 'SiC_V', 'Si_I_100', 'Si_I_110'
params.ml_params['target'] = 'Si_V'
# Transfer prediction types include 'none' (valid for Si_V and SiC_V),
# 'large' (vaild for all targets), 'other defects' (valid for Si_V)
params.ml_params['transfer_type'] = 'none'
params.ml_params['loss_function'] = 'HellingerLoss'
_para = additional_para[(params.ml_params['target'], params.ml_params['transfer_type'])]
params.dftb_params['sigma'] = 0.09
params.dftb_params['with_periodic'] = True
params.dftb_params['siband'] = _para[3]
params.dftb_params['path_to_skf'] = _para[2]
if params.ml_params['transfer_type'] == 'large':
    params.dftb_params['maxiter'] = 1
dataset_train = _para[0]
dataset_test = _para[1]
points = _para[4]


class SiliconDataset(Dataset):
    """Build training and testing dataset."""

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


def prepare_data():
    """Initialize parameters."""
    try:
        os.mkdir('./result')
    except FileExistsError:
        pass

    # Build a random index to select geometries for training and testing
    if params.ml_params['transfer_type'] == 'none':
        rand_index = torch.randperm(80)
        test_index = rand_index[size_train: size_train + size_test].tolist()
    else:
        rand_index = torch.arange(80)
        test_index = rand_index[: size_test].tolist()
    train_index = rand_index[:size_train].tolist()
    f = open('./result/rand_index.dat', 'w')
    np.savetxt(f, rand_index.detach())
    f.close()

    # Read geometries from dataset
    numbers_train, positions_train, latvecs_train, data_train = LoadHdf.load_reference_si(
        dataset_train, train_index, ['homo_lumo', 'eigenvalue'], periodic=True)
    numbers_test, positions_test, latvecs_test, data_test = LoadHdf.load_reference_si(
        dataset_test, test_index, ['homo_lumo', 'eigenvalue'], periodic=True)

    # Build datasets
    train_dataset = SiliconDataset(pack(numbers_train), pack(positions_train),
                                   pack(latvecs_train), data_train['homo_lumo'],
                                   data_train['eigenvalue'])
    test_dataset = SiliconDataset(pack(numbers_test), pack(positions_test),
                                  pack(latvecs_test), data_test['homo_lumo'],
                                  data_test['eigenvalue'])

    # Build objects for DFTB calculation
    system_train = System(pack(numbers_train), pack(positions_train),
                          pack(latvecs_train), siband=params.dftb_params['siband'])
    system_test = System(pack(numbers_test), pack(positions_test),
                         pack(latvecs_test), siband=params.dftb_params['siband'])
    sk_grad = IntegralGenerator.from_dir(
            params.dftb_params['path_to_skf'], system_train, repulsive=False,
            interpolation='spline', sk_type='h5py', with_variable=True,
            siband=params.dftb_params['siband'])
    sk_origin = IntegralGenerator.from_dir(
            params.dftb_params['path_to_skf'], system_test,
            repulsive=False, sk_type='h5py',
            siband=params.dftb_params['siband'])

    # Calculate reference dos and implement sampling
    ref_ev = data_train['eigenvalue']
    energies = torch.linspace(-18, 5, 500).repeat(size_train, 1)
    dos_ref = dos((ref_ev), energies,
                  params.dftb_params['sigma'])

    # Plot training data
    dos_ref_mean = dos_ref.mean(dim=0)
    dos_ref_std = dos_ref.std(dim=0)
    plt.plot(energies[0], dos_ref_mean, '-', linewidth=1.0)
    plt.fill_between(energies[0], dos_ref_mean + dos_ref_std,
                     dos_ref_mean - dos_ref_std, alpha=0.5, facecolor='darkred')
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on', zorder=10)
    plt.xlim((-18.2, 5.2))
    plt.ylim((-1, 70))
    plt.xlabel("Energy [eV]", fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.savefig('./result/ref_dos_cal.png', dpi=500)
    plt.show()

    # Sampling
    fermi_train = data_train['homo_lumo'].mean(dim=1)
    homo_train = data_train['homo_lumo'][..., 0]
    align_train = fermi_train if params.dftb_params['siband'] else homo_train
    energies_train = points.unsqueeze(0).repeat_interleave(
        size_train, 0) + align_train.unsqueeze(-1)

    dos_train = dos((ref_ev), energies_train,
                    params.dftb_params['sigma'])
    data_train_dos = torch.cat((energies_train.unsqueeze(-1),
                                dos_train.unsqueeze(-1)), -1)

    # Plot training range
    plt.plot(energies[0], dos_ref_mean, zorder=0)
    plt.fill_between(energies_train.mean(dim=0), -1, 80, alpha=0.2,
                     facecolor='darkred', zorder=0)
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on', zorder=10)
    plt.xlim((-18.2, 5.2))
    plt.ylim((-1, 70))
    plt.xlabel("Energy [eV]", fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.savefig('./result/training_range.png', dpi=500)
    plt.show()

    return train_dataset, test_dataset, sk_grad, sk_origin, data_train_dos


def data_split(rank, world_size, dataset, batch_size=1, pin_memory=False,
               num_workers=0):
    """Prepare the training data for distributed environment."""
    dataset = dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


def dftb_result(number, position, latvec, sk, **kwargs):
    """Carry out original DFTB calculations."""
    with_grad = kwargs.get('with_grad', False)
    pred = kwargs.get('pred', False)
    dftb = kwargs.get('dftb', False)
    idx = kwargs.get('idx', None)
    sys = System(number, position, latvec,
                 siband=params.dftb_params['siband'])
    cutoff = torch.tensor([18.0]) if params.dftb_params['siband'] else torch.tensor([9.98])
    periodic = Periodic(sys, sys.cell, cutoff=cutoff)
    coulomb = Coulomb(sys, periodic)

    if with_grad:
        skt = SKT(sys, sk, periodic, with_variable=True,
                  fix_onsite=True, fix_U=True)
    else:
        skt = SKT(sys, sk, periodic)

    scc = Scc(sys, skt, params, coulomb, periodic)

    if pred:
        f = open('./result/test/Pred_overlap' + str(int(idx) + 1) + '.dat', 'w')
        np.savetxt(f, skt.S[0].detach())
        f.close()
        f = open('./result/test/Pred_H' + str(int(idx) + 1) + '.dat', 'w')
        np.savetxt(f, skt.H[0].detach())
        f.close()

    if dftb:
        f = open('./result/test/dftb_overlap' + str(int(idx) + 1) + '.dat', 'w')
        np.savetxt(f, skt.S[0])
        f.close()
        f = open('./result/test/dftb_H' + str(int(idx) + 1) + '.dat', 'w')
        np.savetxt(f, skt.H[0])
        f.close()

    return scc


def loss_fn(results, ref_dos, ibatch):
    """Calculate loss during training."""
    loss = 0.
    # Get type of loss function.
    if params.ml_params['loss_function'] == 'MSELoss':
        criterion = torch.nn.MSELoss(reduction='mean')
    elif params.ml_params['loss_function'] == 'L1Loss':
        criterion = torch.nn.L1Loss(reduction='sum')
    elif params.ml_params['loss_function'] == 'KLDivLoss':
        criterion = torch.nn.KLDivLoss(reduction='sum')
    elif params.ml_params['loss_function'] == 'HellingerLoss':
        criterion = tb_math.HellingerLoss()

    # Calculate loss
    ref = ref_dos[..., 1][ibatch]
    fermi_dftb = results.homo_lumo.mean(dim=1)
    homo_dftb = results.homo_lumo[..., 0]
    align_dftb = fermi_dftb if params.dftb_params['siband'] else homo_dftb
    energies_dftb = points + align_dftb
    dos_dftb = dos((results.eigenvalue),
                   energies_dftb, results.params.dftb_params['sigma'])
    loss = loss + criterion(dos_dftb, ref)

    return loss


def main(rank, world_size, train_dataset,
         sk_grad, data_train_dos):
    """ML training to optimize DFTB H and S matrix."""
    # Initialize training parameters
    try:
        os.mkdir('./result/test')
    except FileExistsError:
        pass
    try:
        os.mkdir('./result/abcd')
    except FileExistsError:
        pass

    train_data = data_split(rank, world_size, train_dataset)
    variable = sk_grad.sktable_dict['variable']
    if type(variable) is torch.Tensor:
        variable = Variable(variable, requires_grad=True)
    elif type(variable) is list:
        variable = variable
    if params.ml_params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(variable, lr=params.ml_params['lr'])
    elif params.ml_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(variable, lr=params.ml_params['lr'])
    loss_list = []
    loss_list.append(0)
    time1 = time.time()

    # Training
    for istep in range(params.ml_params['steps']):
        _loss = 0
        print('istep', istep)
        train_data.sampler.set_epoch(istep)
        for ibatch, data in enumerate(train_data):
            scc = dftb_result(data['number'], data['position'],
                              data['latvec'], sk_grad, with_grad=True)
            loss = loss_fn(scc, data_train_dos, data['idx'])
            _loss = _loss + loss
        optimizer.zero_grad()
        _loss.retain_grad()
        _loss.backward(retain_graph=True)
        optimizer.step()
        print("loss:", _loss)
        loss_list.append(_loss.detach())

    time2 = time.time()
    print("time:", time2 - time1)

    # plot loss
    steps = params.ml_params['steps']
    plt.plot(np.linspace(1, steps, steps), loss_list[1:])
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on', zorder=10)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()
    f = open('./result/loss.dat', 'w')
    np.savetxt(f, torch.tensor(loss_list[1:]))
    f.close()


def test(rank, world_size, sk_grad, sk_origin, test_dataset):
    """Test the trained model."""
    test_data = data_split(rank, world_size, test_dataset)
    dos_pred_tot = []
    hl_pred_tot = []
    dos_dftb_tot = []
    hl_dftb_tot = []
    for ibatch, data in enumerate(test_data):
        scc_pred = dftb_result(data['number'], data['position'],
                               data['latvec'], sk_grad, with_grad=True, pred=True,
                               dftb=False, idx=data['idx'])

        # pred
        hl_pred = scc_pred.homo_lumo.detach()
        hl_pred_tot.append(hl_pred)
        dos_pred = scc_pred.dos.detach()
        dos_pred_tot.append(dos_pred)
        dos_energy = scc_pred.dos_energy.detach()
        eigval_pred = scc_pred.eigenvalue.detach()
        eigvec_pred = scc_pred.eigenvector.detach()
        f = open('./result/test/Pred_homo_lumo' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, hl_pred)
        f.close()
        pred_dos = torch.cat((dos_energy.unsqueeze(-1),
                              dos_pred.unsqueeze(-1).squeeze(0)), -1)
        f = open('./result/test/Pred_dos' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, pred_dos)
        f.close()
        f = open('./result/test/Pred_eigenvalue' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigval_pred.squeeze(0))
        f.close()
        f = open('./result/test/Pred_eigenvector' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigvec_pred.squeeze(0))
        f.close()

    for ibatch, data in enumerate(test_data):
        scc_dftb = dftb_result(data['number'], data['position'],
                               data['latvec'], sk_origin, with_grad=False,
                               pred=False, dftb=True, idx=data['idx'])
        # dftb
        hl_dftb = scc_dftb.homo_lumo
        hl_dftb_tot.append(hl_dftb)
        # charge_dftb = scc_dftb.charge
        dos_dftb = scc_dftb.dos
        dos_dftb_tot.append(dos_dftb)
        dos_energy = scc_dftb.dos_energy
        eigval_dftb = scc_dftb.eigenvalue
        eigvec_dftb = scc_dftb.eigenvector
        f = open('./result/test/dftb_homo_lumo' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, hl_dftb)
        f.close()
        dftb_dos = torch.cat((dos_energy.unsqueeze(-1),
                              dos_dftb.unsqueeze(-1).squeeze(0)), -1)
        f = open('./result/test/dftb_dos' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, dftb_dos)
        f.close()
        f = open('./result/test/dftb_eigenvalue' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigval_dftb.squeeze(0))
        f.close()
        f = open('./result/test/dftb_eigenvector' + str(ibatch + 1) + '.dat', 'w')
        np.savetxt(f, eigvec_dftb.squeeze(0))
        f.close()

    # pred_mean
    dos_pred_mean = pack(dos_pred_tot).mean(dim=0)
    dos_pred_std = pack(dos_pred_tot).std(dim=0)
    f = open('./result/Pred_dos_mean.dat', 'w')
    np.savetxt(f, torch.cat((dos_energy.unsqueeze(-1),
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

    # dftb_mean
    dos_dftb_mean = pack(dos_dftb_tot).mean(dim=0)
    dos_dftb_std = pack(dos_dftb_tot).std(dim=0)
    f = open('./result/dftb_dos_mean.dat', 'w')
    np.savetxt(f, torch.cat((dos_energy.unsqueeze(-1),
                             dos_dftb_mean.unsqueeze(-1).squeeze(0)), -1))
    f.close()
    f = open('./result/dftb_dos_std.dat', 'w')
    np.savetxt(f, dos_dftb_std.squeeze(0))
    f.close()
    f = open('./result/dftb_homo_lumo_mean.dat', 'w')
    np.savetxt(f, pack(hl_dftb_tot).mean(dim=0))
    f.close()
    f = open('./result/dftb_homo_lumo_std.dat', 'w')
    np.savetxt(f, pack(hl_dftb_tot).std(dim=0))
    f.close()

    # plot figures
    plt.plot(dos_energy, dos_dftb_mean.squeeze(0), '-', linewidth=1.0)
    plt.fill_between(dos_energy, dos_dftb_mean.squeeze(0) + dos_dftb_std.squeeze(0),
                     dos_dftb_mean.squeeze(0) - dos_dftb_std.squeeze(0),
                     alpha=0.5, facecolor='darkred')
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on', zorder=10)
    plt.xlim((-16, 5.2))
    plt.ylim((-1, 60))
    plt.title('before training')
    plt.xlabel("Energy [eV]", fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.savefig('./result/before_dos.png', dpi=500)
    plt.show()

    plt.plot(dos_energy, dos_pred_mean.squeeze(0), linewidth=1.0)
    plt.fill_between(dos_energy, dos_pred_mean.squeeze(0) + dos_pred_std.squeeze(0),
                     dos_pred_mean.squeeze(0) - dos_pred_std.squeeze(0),
                     alpha=0.5, facecolor='darkred')
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on', zorder=10)
    plt.xlim((-16, 5.2))
    plt.ylim((-1, 60))
    plt.title('after training')
    plt.xlabel("Energy [eV]", fontsize=14)
    plt.ylabel("DOS", fontsize=14)
    plt.savefig('./result/after_dos.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    train_dataset, test_dataset, sk_grad, sk_origin, data_train_dos = prepare_data()
    main(0, 1, train_dataset, sk_grad, data_train_dos)
    print(sk_grad.sktable_dict['variable'], file=open('./result/abcd/abcd.txt', "w"))
    if params.ml_params['test']:
        test(0, 1, sk_grad, sk_origin, test_dataset)
