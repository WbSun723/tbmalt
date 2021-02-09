

class ConstParams:
    """Constant parameters."""

    def __init__(self):
        pass


class Parameter:
    """DFTB parameters."""

    def __init__(self):
        self.mix = 'Anderson'  # -> Anderson, Simple
        self.scc = 'scc'  # 'scc', 'nonscc', 'xlbomd'
        self.maxiter = 60  # -> max SCC loop

    @classmethod
    def get_ml_params(cls):
        """Return machine learning parameters."""
        return {'lr': 0.1,

                # training steps
                'steps': 3,

                # get loss function type
                'loss_function': 'MSELoss',

                # get optimizer
                'optimizer': 'SCG'}

    def training():
        pass
