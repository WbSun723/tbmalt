"""Train code."""
import torch
import numpy as np
from torch.autograd import Variable
from tbmalt.common.parameter import Parameter
from tbmalt.tb.sk import SKT
from tbmalt.common.batch import pack
from tbmalt.tb.dftb.scc import Scc
from tbmalt.io.loadskf import IntegralGenerator
import matplotlib.pyplot as plt
from tbmalt.ml.feature import Dscribe
from tbmalt.ml.scikitlearn import SciKitLearn


class Train:
    """Train class."""

    def __init__(self, system, reference, variable, parameter, **kwargs):
        self.system = system
        self.reference = reference
        self.periodic = kwargs.get('periodic', None)
        self.coulomb = kwargs.get('coulomb', None)

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
                results.dipole, self.reference['dipole'])
        if 'charge' in self.properties:
            self.loss = self.loss + self.criterion(
                results.charge, self.reference['charge'])
        if 'gap' in self.properties:
            homolumo = results.homo_lumo
            refhl = pack(self.reference['homo_lumo'])
            gap = homolumo[:, 1] - homolumo[:, 0]
            refgap = refhl[:, 1] - refhl[:, 0]
            self.loss = self.loss + self.criterion(gap, refgap)
        if 'cpa' in self.properties:
            cpa = results.cpa
            refcpa = self.reference['hirshfeld_volume_ratio']
            self.loss = self.loss + self.criterion(cpa, refcpa)

    def __predict__(self, system):
        """Predict with training results."""
        pass

    def __plot__(self, steps, loss, **kwargs):
        """Visualize training results."""
        compression_radii = kwargs.get('compression_radii', None)

        # plot loss
        plt.plot(np.linspace(1, steps, steps), loss)
        plt.show()

        # plot compression radii
        if compression_radii is not None:
            compr = pack(compression_radii)
            for ii in range(compr.shape[1]):
                for jj in range(compr.shape[2]):
                    plt.plot(np.linspace(1, steps, steps), compr[:, ii, jj])
            plt.show()


class Integal(Train):
    """Optimize integrals."""

    def __init__(self, system, reference, parameter, **kwargs):
        """Initialize parameters."""
        self.sk = IntegralGenerator.from_dir(
            parameter.dftb_params['path_to_skf'], system, repulsive=False,
            interpolation='spline', sk_type='h5py', with_variable=True)
        self.ml_variable = self.sk.sktable_dict['variable']
        super().__init__(system, reference, self.ml_variable, parameter, **kwargs)

    def __call__(self, target, plot=True):
        """Train spline parameters with target properties."""
        super().__call__(target)
        self._loss = []
        for istep in range(self.steps):
            self._update_train()
            self._loss.append(self.loss.detach())

        if plot:
            super().__plot__(self.steps, self._loss)

    def _update_train(self):
        skt = SKT(self.system, self.sk, self.periodic, with_variable=True,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, self.coulomb, self.properties)
        super().__loss__(scc)
        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()

    def predict(self, system, coulomb=None):
        """Predict with optimized Hamiltonian and overlap."""
        skt = SKT(system, self.sk, with_variable=True,
                  fix_onsite=True, fix_U=True)
        return Scc(system, skt, self.params, coulomb, self.properties)


class CompressionRadii(Train):
    """Optimize compression radii."""

    def __init__(self, system, reference, parameter, **kwargs):
        """Initialize parameters."""
        self.nbatch = system.size_batch
        self.ml_variable = Variable(torch.ones(
            self.nbatch, max(system.size_system)) * 3.5, requires_grad=True)
        self.sk = IntegralGenerator.from_dir(
            parameter.dftb_params['path_to_skf'], system, repulsive=False,
            sk_type='h5py', homo=False, interpolation='bicubic_interpolation',
            compression_radii_grid=parameter.ml_params['compression_radii_grid'])
        super().__init__(system, reference, [self.ml_variable], parameter, **kwargs)

    def __call__(self, target, plot=True):
        """Train compression radii with target properties."""
        super().__call__(target)
        self._loss, self._compr = [], []
        for istep in range(self.steps):
            self._update_train()

        if plot:
            super().__plot__(self.steps, self._loss, compression_radii=self._compr)

    def _update_train(self):
        skt = SKT(self.system, self.sk, self.periodic, compression_radii=self.ml_variable,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, self.coulomb, self.properties)
        super().__loss__(scc)
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

    def predict(self, system):
        """Predict with optimized Hamiltonian and overlap."""
        feature_type = 'acsf'
        targets = self.ml_variable
        feature = Dscribe(self.system, feature_type=feature_type).features
        feature_pred = Dscribe(system, feature_type=feature_type).features
        split_ratio = 0.5
        predict_compr = SciKitLearn(
            self.system, feature, targets.detach(), system_pred=system,
            feature_pred=feature_pred, ml_method=self.params.ml_params['ml_method'],
            split_ratio=split_ratio).prediction
        predict_compr = torch.clamp(
            predict_compr, self.params.ml_params['compression_radii_min'],
            self.params.ml_params['compression_radii_max'])
        print('predicted compr', predict_compr)
        skt = SKT(system, self.sk, compression_radii=predict_compr,
                  fix_onsite=True, fix_U=True)
        return Scc(system, skt, self.params, self.properties)


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
