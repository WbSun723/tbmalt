"""Train code."""
import torch
from torch.autograd import Variable
from tbmalt.common.parameter import Parameter
from tbmalt.tb.sk import SKT
from tbmalt.tb.dftb.scc import Scc


class Train():

    def __init__(self, system, reference, variable, parameter, **kwargs):
        self.system = system
        self.reference = reference
        self.variable = Variable(variable, requires_grad=True)

        self.params = Parameter if parameter is None else parameter

        self.ml_params = self.params.get_ml_params()
        self.lr = self.ml_params['lr']

        # get loss function type
        if self.ml_params['loss_function'] == 'MSELoss':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif self.ml_params['loss_function'] == 'L1Loss':
            self.criterion = torch.nn.L1Loss(reduction='sum')

        # get optimizer
        if self.ml_params['optimizer'] == 'SCG':
            self.optimizer = torch.optim.SGD([self.variable], lr=self.lr)
        elif self.ml_params['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam([self.variable], lr=self.lr)

    def __call__(self, properties):
        self.properties = properties
        self.steps = self.ml_params['steps']

    def __loss__(self, train_property):
        self.loss = 0
        if 'dipole' in self.properties:
            self.loss = self.criterion(train_property.properties.dipole,
                                       self.reference['dipole'])


class Integal(Train):

    def __init__(self, system, reference, variable, parameter, **kwargs):
        pass


class CompressionRadii(Train):
    """Train compression radii in basis functions."""

    def __init__(self, system, reference, variable, parameter, sk, **kwargs):
        super().__init__(system, reference, variable, parameter, **kwargs)
        self.sk = sk

    def __call__(self, properties):
        super().__call__(properties)
        for istep in range(self.steps):
            self._update_train()

    def _update_train(self):
        skt = SKT(self.system, self.sk, compression_radii=self.variable,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, properties=self.properties)
        super().__loss__(scc)
        print('self.variable', self.variable)
        print('loss', self.loss)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
