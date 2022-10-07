"""Example to run training by multiprocessing."""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.distributed.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
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

# Set general parameters for training and DFTB calculations
torch.set_num_threads(6)
torch.set_printoptions(6, profile='full')
torch.set_default_dtype(torch.float64)
size_train, size_test = 10, 20
params = Parameter(ml_params=True)
params.ml_params['task'] = 'mlIntegral'
params.ml_params['optimizer'] = 'Adam'
params.ml_params['steps'] = 2000
params.ml_params['lr'] = 0.000005
params.ml_params['test'] = True
# Taget systems include 'Si_V', 'SiC_V'
params.ml_params['target'] = 'Si_V'

# Transfer prediction types include 'none' (valid for Si_V and SiC_V),
# and 'other defects' (valid for Si_V)
params.ml_params['transfer_type'] = 'none'
params.ml_params['loss_function'] = 'HellingerLoss'
_para = additional_para[(params.ml_params['target'], params.ml_params['transfer_type'])]
params.dftb_params['sigma'] = 0.09
params.dftb_params['with_periodic'] = True
params.dftb_params['siband'] = _para[3]
params.dftb_params['path_to_skf'] = _para[2]
dataset_train = _para[0]
dataset_test = _para[1]
points = _para[4]


class SiliconDataset(Dataset):
    """Build training dataset."""

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


def prepare_data(rand_index):
    """Read training and testing data to generate dataset."""
    try:
        os.mkdir('./result')
    except FileExistsError:
        pass

    # Build a random index to select geometries for training and testing
    train_index = rand_index[: size_train].tolist()
    test_index = rand_index[60: 60 + size_test].tolist()
    f = open('./result/rand_index.dat', 'w')
    np.savetxt(f, rand_index.detach())
    f.close()

    # Read geometry
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

    # Build sk objects
    system_train = System(pack(numbers_train), pack(positions_train),
                          pack(latvecs_train), siband=params.dftb_params['siband'])
    system_test = System(pack(numbers_test), pack(positions_test),
                         pack(latvecs_test), siband=params.dftb_params['siband'])

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
    plt.fill_between(energies[0], dos_ref_mean + dos_ref_std, dos_ref_mean - dos_ref_std,
                     alpha=0.5, facecolor='darkred')
    plt.tick_params(direction='in', labelsize='13', width=1.1, top='on',
                    right='on', zorder=10)
    plt.xlim((-18.2, 5.2))
    plt.ylim((-1, 70))
    plt.xlabel("Energy [eV]")
    plt.ylabel("DOS")
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

    return train_dataset, test_dataset, system_train, system_test, sk_origin, data_train_dos


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


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
        criterion = torch.nn.MSELoss(reduction='sum')
    elif params.ml_params['loss_function'] == 'L1Loss':
        criterion = torch.nn.L1Loss(reduction='sum')
    elif params.ml_params['loss_function'] == 'KLDivLoss':
        criterion = torch.nn.KLDivLoss(reduction='sum')
    elif params.ml_params['loss_function'] == 'HellingerLoss':
        criterion = tb_math.HellingerLoss()

    # Calculate loss
    ref = ref_dos[..., 1][ibatch]
    loss = loss + criterion(results, ref)
    return loss


class DFTB_DDP(torch.nn.Module):
    """Implement DFTB calculation within the framework of nn.Module."""

    def __init__(self, path, system):
        super(DFTB_DDP, self).__init__()
        self.sk_grad = IntegralGenerator.from_dir(
            path, system, repulsive=False,
            interpolation='spline', sk_type='h5py', with_variable=True,
            siband=params.dftb_params['siband'])
        if params.dftb_params['siband']:
            self.parameters = torch.nn.ParameterList([torch.nn.Parameter(ivar)
                                                      for ivar in
                                                      self.sk_grad.sktable_dict['variable']])
        else:
            _idx = torch.tensor([10, 11, 12, 13, 16, 17, 18, 19])
            self.index_grad = torch.cat([(_idx + ii * 20) for ii in range(4)])
            self.parameters = torch.nn.ParameterList(
                [torch.nn.Parameter(self.sk_grad.sktable_dict['variable'][ii])
                 for ii in self.index_grad])

    def forward(self, data, path, system):
        if params.dftb_params['siband']:
            _sk_pred = IntegralGenerator.from_dir(
                path, system, repulsive=False, interpolation='spline',
                sk_type='h5py', siband=params.dftb_params['siband'],
                pred=list(self.parameters()))
        else:
            _para = torch.zeros_like(self.sk_grad.sktable_dict['variable'][0]
                                     ).unsqueeze(dim=0).repeat_interleave(80, dim=0)
            _para[self.index_grad] = pack(list(self.parameters()))
            self.list_update = list(_para)
            _sk_pred = IntegralGenerator.from_dir(
                path, system, repulsive=False, interpolation='spline',
                sk_type='h5py', pred=self.list_update)
        scc = dftb_result(data['number'], data['position'],
                          data['latvec'], _sk_pred, with_grad=True)
        fermi_dftb = scc.homo_lumo.mean(dim=1)
        homo_dftb = scc.homo_lumo[..., 0]
        align_dftb = fermi_dftb if params.dftb_params['siband'] else homo_dftb
        energies_dftb = points + align_dftb
        dos_dftb = dos((scc.eigenvalue),
                       energies_dftb, 0.09)
        return dos_dftb


def main(rank, world_size, rand_index):
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

    setup(rank, world_size)
    train_dataset, test_dataset, system_train, system_test, sk_origin, data_train_dos =\
        prepare_data(rand_index)
    train_data = data_split(rank, world_size, train_dataset)
    device = torch.device("cpu")
    model = DFTB_DDP(params.dftb_params['path_to_skf'],
                     system_train)
    ddp_model = DDP(model.to(device))

    if params.ml_params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.ml_params['lr'])
    elif params.ml_params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.ml_params['lr'])
    loss_list = []
    loss_list.append(0)

    # Training
    for istep in range(params.ml_params['steps']):
        print('istep', istep)
        train_data.sampler.set_epoch(istep)
        for ibatch, data in enumerate(train_data):
            dos_dftb = ddp_model(data, params.dftb_params['path_to_skf'],
                                 system_train)
            loss = loss_fn(dos_dftb, data_train_dos, data['idx'])
            print("rank:", int(rank), "loss:", loss, "idx", data['idx'])
            optimizer.zero_grad()
            loss.retain_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.detach())

    f = open('./result/loss' + str(int(rank)) + '.dat', "w")
    np.savetxt(f, torch.tensor(loss_list[1:]))
    f.close()

    if rank == 0:
        update_para = list(model.parameters()) if params.dftb_params['siband'] else\
            list(model.list_update)
        print(update_para, file=open('./result/abcd/abcd.txt', "w"))

        if params.ml_params['test']:
            sk_grad = IntegralGenerator.from_dir(
                    params.dftb_params['path_to_skf'], system_test, repulsive=False,
                    interpolation='spline', sk_type='h5py',
                    siband=params.dftb_params['siband'],
                    pred=update_para)
            test(0, 1, sk_grad, sk_origin, test_dataset)


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
    plt.xlim((-16, 5.2))
    plt.ylim((-1, 60))
    plt.title('before training')
    plt.xlabel("Energy [eV]")
    plt.ylabel("DOS")
    plt.savefig('./result/before_dos.png', dpi=500)
    plt.show()

    plt.plot(dos_energy, dos_pred_mean.squeeze(0), linewidth=1.0)
    plt.fill_between(dos_energy, dos_pred_mean.squeeze(0) + dos_pred_std.squeeze(0),
                     dos_pred_mean.squeeze(0) - dos_pred_std.squeeze(0),
                     alpha=0.5, facecolor='darkred')
    plt.xlim((-16, 5.2))
    plt.ylim((-1, 60))
    plt.title('after training')
    plt.xlabel("Energy [eV]")
    plt.ylabel("DOS")
    plt.savefig('./result/after_dos.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    world_size = 10
    rand_index = torch.randperm(80)
    time1 = time.time()
    mp.spawn(main, args=(world_size, rand_index),
             nprocs=world_size)
    time2 = time.time()
    print("time:", time2 - time1)
