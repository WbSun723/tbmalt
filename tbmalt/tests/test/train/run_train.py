import torch
import os
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.ml.train import Integal, CompressionRadii
from tbmalt.common.parameter import Parameter
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.tb.sk import SKT
from tbmalt.common.structures.periodic import Periodic
from tbmalt.tb.coulomb import Coulomb
from tbmalt.tb.dftb.scc import Scc
from tbmalt.ml.feature import Dscribe
from tbmalt.ml.scikitlearn import SciKitLearn
import matplotlib.pyplot as plt

torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)
# os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')
os.system('cp /home/gz_fan/Public/tbmalt/reference.hdf .')
os.system('cp -r /home/gz_fan/Public/tbmalt/dataset .')
size_train, size_test = 20, 1000
params = Parameter(ml_params=True)
params.ml_params['task'] = 'mlIntegral'
params.ml_params['steps'] = 10
params.ml_params['target'] = ['charge']
params.ml_params['ml_method'] = 'linear'  # nn, linear, forest
dataset = './dataset/aims_6000_01.hdf'
dataset_dftb = './dataset/scc_6000_01.hdf'


def train(parameter=None, ml=None):
    """Initialize parameters."""
    if params.ml_params['task'] == 'mlIntegral':
        params.dftb_params['path_to_skf'] = './slko/skf.hdf.init2'
        params.ml_params['lr'] = 0.0018
    elif params.ml_params['task'] == 'mlCompressionR':
        params.dftb_params['path_to_skf'] = './slko/skf.hdf.compr'
        params.ml_params['lr'] = 0.02
        params.ml_params['compression_radii_grid'] = torch.tensor([
            01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
            04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    latvec = torch.tensor([[6., 0., 0.], [0., 6., 0.], [0., 0., 6.]])

    numbers_train, positions_train, data_train = LoadHdf.load_reference(
        dataset, size_train, ['charge', 'dipole', 'homo_lumo', 'hirshfeld_volume_ratio'])
    sys_train = System(numbers_train, positions_train, latvec.repeat(len(numbers_train), 1, 1))
    periodic_train = Periodic(sys_train, sys_train.cell, cutoff=10.0, unit='bohr')
    coulomb_train = Coulomb(sys_train, periodic_train)

    numbers_test, positions_test, data_test = LoadHdf.load_reference(
        dataset, size_test, ['charge', 'dipole', 'homo_lumo', 'hirshfeld_volume_ratio'])

    numbers_dftb, positions_dftb, data_dftb = LoadHdf.load_reference(
        dataset_dftb, size_test, ['charge', 'dipole', 'homo_lumo'])

    sys_test = System(numbers_test, positions_test, latvec.repeat(len(numbers_test), 1, 1))
    periodic_test = Periodic(sys_test, sys_test.cell, cutoff=10.0, unit='bohr')
    coulomb_test = Coulomb(sys_test, periodic_test)

    # optimize integrals directly
    if params.ml_params['task'] == 'mlIntegral':
        integral = Integal(sys_train, data_train, params, periodic=periodic_train, coulomb=coulomb_train)
        integral(params.ml_params['target'])

        scc_pred = integral.predict(sys_test, coulomb_test, periodic_test)

    elif params.ml_params['task'] == 'mlCompressionR':
        compr = CompressionRadii(sys_train, data_train, params, periodic=periodic_train, coulomb=coulomb_train)
        compr(params.ml_params['target'])

        scc_pred = compr.predict(sys_test)

    print('property:', 'charge')
    print(abs((scc_pred.charge - data_test['charge'])).sum())
    print(abs((data_dftb['charge'] - data_test['charge'])).sum())
    print(scc_pred.charge, '\n', data_dftb['charge'])
    heavy_mask = data_test['charge'].gt(1)
    heavy_qzero = torch.floor(data_test['charge'][heavy_mask])
    xx = torch.linspace(1, heavy_qzero.shape[0], heavy_qzero.shape[0])
    # plt.plot(data_test['charge'][heavy_mask] - heavy_qzero,
    #          scc_pred.charge.detach()[heavy_mask] - heavy_qzero, 'xr')
    plt.plot(xx, scc_pred.charge.detach()[heavy_mask] - heavy_qzero, 'xr', label='predict')
    # plt.plot(xx, data_dftb['charge'][heavy_mask] - heavy_qzero, 'vb', label='previous parameters')
    plt.legend()
    plt.show()
    # if 'dipole' in params.ml_params['target']:
    print('property:', 'dipole')
    print(abs((scc_pred.dipole - data_test['dipole'])).sum())
    print(abs((data_dftb['dipole'] - data_test['dipole'])).sum())
    plt.plot(data_test['dipole'], scc_pred.dipole.detach(), 'xr', label='predict')
    plt.plot(data_test['dipole'], data_dftb['dipole'], 'vb', label='previous parameters')
    plt.plot(torch.linspace(-0.8, 0.8, 10), torch.linspace(-0.8, 0.8, 10), color='k')
    plt.legend()
    plt.show()

    if 'cpa' in params.ml_params['target']:
        sys_dftb = System(numbers_dftb, positions_dftb)
        sk = IntegralGenerator.from_dir('./slko/skf.hdf.init', sys_dftb, sk_type='h5py')
        skt = SKT(sys_dftb, sk)
        scc = Scc(sys_dftb, skt, params, ['cpa'])
        print('property:', 'cpa')
        print(abs((scc_pred.cpa - data_test['hirshfeld_volume_ratio'])).sum())
        print(abs((scc.cpa - data_test['hirshfeld_volume_ratio'])).sum())


def test():
    """Test repulsive of hdf."""
    # -> define all input parameters
    properties = ['dipole', 'charge', 'homo_lumo', 'energy',
                  'formation_energy']
    reference_size = 1000
    split_ratio = 0.5
    ml_method = 'nn'  # forest, linear, nn, svm, grad_boost
    # _feature = ['U', 'valence', 'atom_radii_emp', 'atom_radii_cal',
    #             'ionization_energy', 'electronegativity', 'electron_affinity',
    #             'l_number']
    _feature = ['U', 'valence', 'atom_radii_emp', 'atom_radii_cal',
                'electronegativity', 'l_number']
    feature_type = 'acsf'
    parameter = Parameter()

    # load the the generated dataset
    numbers, positions, data_nonscc = LoadHdf.load_reference(
        './dataset/nonscc_6000_01.hdf', reference_size, properties)
    # load the the generated dataset
    _, _, data = LoadHdf.load_reference('scc.hdf', reference_size, properties)

    sys = System(numbers, positions)

    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    skt = SKT(molecule, sktable)
    scc = Scc(molecule, skt, parameter, ['dipole', 'charge', 'homo_lumo'])

    targets = data['charge']
    features = Dscribe(sys, feature_type=feature_type,
                       static_parameter=_feature).features
    predict_cha = SciKitLearn(sys, features, targets, ml_method, split_ratio).prediction
    print('predict_cha', predict_cha)
    print('scc', data['charge'])
    parameter.scc = 'xlbomd'
    xlbomd = Scc(molecule, skt, parameter, ['dipole', 'charge', 'homo_lumo'],
                 charge=predict_cha)
    tol1 = (abs(data['dipole'] - scc.properties.dipole)).sum()
    tol2 = (abs(data['dipole'] - xlbomd.properties.dipole)).sum()
    tol3 = (abs(data['dipole'] - data_nonscc['dipole'])).sum()
    print(tol1, tol2, tol3, tol2 / tol3)

    tol1 = (abs(data['charge'] - scc.charge)).sum()
    tol2 = (abs(data['charge'] - xlbomd.charge)).sum()
    tol3 = (abs(data['charge'] - data_nonscc['charge'])).sum()
    print(tol1, tol2, tol3, tol2 / tol3)


def xlbomd():
    """Test."""
    # -> define all input parameters
    properties = ['dipole', 'charge']
    reference_size = 1000
    params = Parameter()

    # load the the generated dataset
    numbers, positions, data_nonscc = LoadHdf.load_reference(
        'nonscc.hdf', reference_size, properties)
    # load the the generated dataset
    _, _, data = LoadHdf.load_reference('scc.hdf', reference_size, properties)

    sys = System(numbers, positions)

    molecule = System(numbers, positions)
    sk = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)

    variable = _get_charge(sys)
    train = CompressionRadii(sys, data, variable, params, sk)
    train(properties)


def _get_charge(system):
    pass


train()
