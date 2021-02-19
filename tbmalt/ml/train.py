"""Train code."""
import torch
from torch.autograd import Variable
from tbmalt.common.parameter import Parameter
from tbmalt.tb.sk import SKT
from tbmalt.common.batch import pack
from tbmalt.tb.dftb.scc import Scc


class Train():
    """Train class."""

    def __init__(self, system, reference, variable, parameter, **kwargs):
        self.system = system
        self.reference = reference
        if type(variable) is torch.Tensor:
            self.variable = Variable(variable, requires_grad=True)
        elif type(variable) is list:
            self.variable = variable  # Variable(pack(variable), requires_grad=True)
        opt = self.variable if type(self.variable) is list else [self.variable]

        self.params = Parameter if parameter is None else parameter
        self.lr = self.params.ml_params['lr']

        # get loss function type
        if self.params.ml_params['loss_function'] == 'MSELoss':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        elif self.params.ml_params['loss_function'] == 'L1Loss':
            self.criterion = torch.nn.L1Loss(reduction='sum')

        # get optimizer
        if self.params.ml_params['optimizer'] == 'SCG':
            self.optimizer = torch.optim.SGD(opt, lr=self.lr)
        elif self.params.ml_params['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(opt, lr=self.lr)

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
            print(results.properties.dipole, self.reference['dipole'])
        if 'charge' in self.properties:
            self.loss = self.loss + self.criterion(
                results.charge, self.reference['charge'])


class Integal(Train):
    """Train integral."""

    def __init__(self, system, reference, variable, parameter, sk, **kwargs):
        super().__init__(system, reference, variable, parameter, **kwargs)
        self.sk = sk

    def __call__(self, properties):
        """Call train class with properties."""
        super().__call__(properties)
        loss = []
        for istep in range(self.steps):
            self._update_train()
            loss.append(self.loss.detach())
        return loss

    def _update_train(self):
        skt = SKT(self.system, self.sk, with_variable=True,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, properties=self.properties)
        super().__loss__(scc)
        print('loss', self.loss, scc.charge)
        print('grad', self.variable[0].grad)

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()


class CompressionRadii(Train):
    """Train compression radii in basis functions."""

    def __init__(self, system, reference, variable, parameter, sk, **kwargs):
        super().__init__(system, reference, variable, parameter, **kwargs)
        self.sk = sk
        self.compr_min = self.params.ml_params['compression_radii_min']
        self.compr_max = self.params.ml_params['compression_radii_max']

    def __call__(self, properties):
        """Call train class with properties."""
        super().__call__(properties)
        self.loss_, self.var = [], []
        for istep in range(self.steps):
            self._update_train()
            print('step', istep)
        return self.loss_, pack(self.var)

    def _update_train(self):
        skt = SKT(self.system, self.sk, compression_radii=self.variable,
                  fix_onsite=True, fix_U=True)
        scc = Scc(self.system, skt, self.params, properties=self.properties)


        print('gamma', scc.gamma.shape)
        self.loss = self.criterion(scc.gamma, torch.ones(*scc.gamma.shape))

        # super().__loss__(scc)
        print('compression radii', self.variable)
        print('gradient', self.variable.grad)
        self.loss_.append(self.loss.detach())
        self.var.append(self.variable.detach().clone())

        self.optimizer.zero_grad()
        self.loss.backward(retain_graph=True)
        self.optimizer.step()

        _compr = self.variable.detach().clone()
        min_mask = _compr[_compr != 0].lt(self.compr_min)
        max_mask = _compr[_compr != 0].gt(self.compr_max)
        if True in min_mask or True in max_mask:
            with torch.no_grad():
                self.variable.clamp_(self.compr_min, self.compr_max)


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
