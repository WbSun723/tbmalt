"""Train code."""
import torch
import numpy as np
from torch.autograd import Variable
from tbmalt.common.parameter import Parameter
from tbmalt.tb.sk import SKT
from tbmalt.common.batch import pack
from tbmalt.tb.dftb.scc import Scc
from tbmalt.common.structures.system import System
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.io.loadskf import IntegralGenerator
import matplotlib.pyplot as plt


def train(parameter=None, ml=None):
    """Initialize parameters."""
    params = Parameter(ml_params=True)
    params.dftb_params['scc'] = 'scc'
    params.ml_params['task'] = 'mlCompressionR'
    params.ml_params['steps'] = 50
    params.ml_params['target'] = ['charge', 'dipole']
    size = 6

    if params.ml_params['task'] == 'mlIntegral':
        params.dftb_params['path_to_skf'] = '../tests/unittests/slko/skf.hdf.init2'
        params.ml_params['lr'] = 0.001
    elif params.ml_params['task'] == 'mlCompressionR':
        params.dftb_params['path_to_skf'] = '../tests/unittests/slko/skf.hdf.compr'
        params.ml_params['lr'] = 0.05
        params.ml_params['compression_radii_grid'] = torch.tensor([
            01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
            04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])

    numbers, positions, data = LoadHdf.load_reference(
        '../tests/unittests/dataset/aims_6000_01.hdf', size, params.ml_params['target'])
    sys = System(numbers, positions)

    # optimize integrals directly
    if params.ml_params['task'] == 'mlIntegral':
        integral = Integal(sys, data, params)
        integral(params.ml_params['target'])
    elif params.ml_params['task'] == 'mlCompressionR':
        compr = CompressionRadii(sys, data, params)
        compr(params.ml_params['target'])
    # print('positions', positions)


class Train:
    """Train class."""

    def __init__(self, system, reference, variable, parameter, **kwargs):
        self.system = system
        self.reference = reference
        if type(variable) is torch.Tensor:
            self.variable = Variable(variable, requires_grad=True)
        elif type(variable) is list:
            self.variable = variable

        self.params = Parameter if parameter is None else parameter
        self.lr = self.params.ml_params['lr']

        # get loss function type
        if self.params.ml_params['loss_function'] == 'MSELoss':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif self.params.ml_params['loss_function'] == 'L1Loss':
            self.criterion = torch.nn.L1Loss(reduction='sum')

        # get optimizer
        if self.params.ml_params['optimizer'] == 'SCG':
            self.optimizer = torch.optim.SGD(self.variable, lr=self.lr)
        elif self.params.ml_params['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.variable, lr=self.lr)

    def __call__(self, properties):
        """Call train class with properties."""
        self.properties = properties
        self.steps = self.params.ml_params['steps']

    def __loss__(self, results):
        """Get loss function."""
        self.loss = 0.
        if 'dipole' in self.properties:
            self.loss = self.loss + self.criterion(
                results.properties.dipole, self.reference['dipole'])
        if 'charge' in self.properties:
            self.loss = self.loss + self.criterion(
                results.charge, self.reference['charge'])


class Integal(Train):
    """Optimize integrals."""

    def __init__(self, system, reference, parameter):
        """Initialize parameters."""
        self.sk = IntegralGenerator.from_dir(
            parameter.dftb_params['path_to_skf'], system, repulsive=False,
            interpolation='spline', sk_type='h5py', with_variable=True)
        self.ml_variable = self.sk.sktable_dict['variable']
        super().__init__(system, reference, self.ml_variable, parameter)

    def __call__(self, target):
        super().__call__(target)
        self._loss = []
        for istep in range(self.steps):
            self._update_train()
            self._loss.append(self.loss.detach())

        plt.plot(np.linspace(1, len(self._loss), len(self._loss)), self._loss)
        plt.show()

    def _update_train(self):
        skt = SKT(self.system, self.sk, with_variable=True,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, self.properties)
        super().__loss__(scc)
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()


class CompressionRadii(Train):
    """Optimize compression radii."""

    def __init__(self, system, reference, parameter):
        """Initialize parameters."""
        self.nbatch = system.size_batch
        self.ml_variable = Variable(torch.ones(
            self.nbatch, max(system.size_system)) * 3.5, requires_grad=True)
        self.sk = IntegralGenerator.from_dir(
            parameter.dftb_params['path_to_skf'], system, repulsive=False,
            sk_type='h5py', homo=False, interpolation='bicubic_interpolation',
            compression_radii_grid=parameter.ml_params['compression_radii_grid'])
        super().__init__(system, reference, [self.ml_variable], parameter)

    def __call__(self, target):
        super().__call__(target)
        self._loss, self._compr = [], []
        for istep in range(self.steps):
            self._update_train()

        steps = len(self._loss)
        plt.plot(np.linspace(1, steps, steps), self._loss)
        plt.show()
        for ii in range(self.ml_variable.shape[0]):
            for jj in range(self.ml_variable.shape[1]):
                plt.plot(np.linspace(1, steps, steps), pack(self._compr)[:, ii, jj])
        plt.show()

    def _update_train(self):
        skt = SKT(self.system, self.sk, compression_radii=self.ml_variable,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, self.properties)
        super().__loss__(scc)
        print('compression radii', self.variable)
        print('gradient', self.ml_variable.grad)
        self._loss.append(self.loss.detach())
        self._compr.append(self.ml_variable.detach().clone())

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self._check(self.params)

    def _check(self, para):
        """Check the machine learning variables each step.

        When training compression radii, sometimes the compression radii will
        be out of range of given grid points and go randomly, therefore here
        the code makes sure the compression radii is in the defined range.
        """
        # detach remove initial graph and make sure compr_ml is leaf tensor
        compr_ml = self.ml_variable.detach().clone()
        min_mask = compr_ml[compr_ml != 0].lt(para.ml_params['compression_radii_min'])
        max_mask = compr_ml[compr_ml != 0].gt(para.ml_params['compression_radii_max'])
        if True in min_mask or True in max_mask:
            with torch.no_grad():
                self.ml_variable.clamp_(para.ml_params['compression_radii_min'],
                                        para.ml_params['compression_radii_max'])


class Charge(Train):
    """Train charge."""

    def __init__(self, system, reference, variable, parameter, sk, **kwargs):
        super().__init__(system, reference, variable, parameter, **kwargs)
        self.sk = sk

    def __call__(self, properties):
        """Call train class with properties."""
        super().__call__(properties)
        for istep in range(self.steps):
            self._update_train()

    def _update_train(self):
        skt = SKT(self.system, self.sk)
        scc = Scc(self.system, skt, self.params, properties=self.properties)


train()
