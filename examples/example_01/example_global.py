"""
Created on Tue May 20 10:39:22 2025

@author: wbsun
"""
import pickle
from os.path import exists
from typing import Any, List
import random
import torch
import h5py
from sklearn.ensemble import RandomForestRegressor
from ase.build import molecule

from tbmalt import Geometry, OrbitalInfo
from tbmalt.ml.module import Calculator
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, VcrSkFeed, SkfOccupationFeed, HubbardFeed
from tbmalt.io.dataset import DataSetIM
from tbmalt.ml.acsf import Acsf
from tbmalt.common.batch import pack
from tbmalt.tools.downloaders import download_dftb_parameter_set

Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)
device = torch.device('cpu')

# ============== #
# STEP 1: Inputs #
# ============== #

# 1.1: System settings
# --------------------

# Provide a list of molecules upon which TBMaLT is to be run
targets = ['dipole']
sources_train = ['run1/train', 'run2/train', 'run3/train']
sources_test = ['run1/test', 'run2/test', 'run3/test']

# Provide information about the orbitals on each atom; this is keyed by atomic
# numbers and valued by azimuthal quantum numbers like so:
#   {Z₁: [ℓᵢ, ℓⱼ, ..., ℓₙ], Z₂: [ℓᵢ, ℓⱼ, ..., ℓₙ], ...}
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

# 1.2: Model settings
# -------------------
# Location at which the DFTB parameter set database is located
parameter_db_path = 'example_dftb_vcr.h5'
parameter_db_path_std = 'example_dftb_parameters.h5'

# Should fitting be performed here?
fit_model = True
pred_model = True

# Number of fitting cycles, number of batch size each cycle
number_of_epochs = 10
n_fit_batch = 10
lr = 0.01
onsite_lr = 1e-3
criterion = getattr(torch.nn, 'MSELoss')(reduction='mean')
# ACSF parameters
g1_params = 6.0
g2_params = torch.tensor([0.5, 1.0])
g4_params = torch.tensor([[0.02, 1.0, -1.0]])
element_resolve = True  # If ACSF is element resolved
n_estimators = 100  # parameters in random forest
global_r = True  # If orbs parameters is global
tolerance = 1e-6  # tolerance of loss
shell_resolved = False  # If DFTB Hubbard U is shell resolved

# Location of a file storing the properties that will be fit to.
target_path = './dataset.h5'


# ============= #
# STEP 2: Setup #
# ============= #

# load data set
def load_target_data(path: str, sources: List[str], targets: List[str]) -> Any:
    """Load fitting target data.

    Arguments:
        path: path to a database in which the fitting data can be found.
        sources: a list of paths specifying the groups from which data
            should be loaded; one for each system.
        targets: paths relative to `source` specifying the HDF5 datasets
                to load.

    Returns:
        targets: returns an <OBJECT> storing the data to which the model is to
            be fitted.
    """
    # Data could be loaded from a json file or an hdf5 file; use your own
    # discretion here. A dictionary might be the best object in which to store
    # the target data.
    with h5py.File(path, 'r') as f:
        _sources = []
        for sou in sources:
            _sources.extend([sou + '/' + i for i in (f[sou].keys())])
    return DataSetIM.load_data(path, _sources, targets)


# 2.1: Target system specific objects
# -----------------------------------
if fit_model or pred_model:
    dataloder_fit, dataloder_test = [], []
    for s_fit, s_test in zip(sources_train, sources_test):
        dataloder_fit.append(load_target_data(target_path, sources_train, targets))
        dataloder_test.append(load_target_data(target_path, sources_test, targets))


# 2.2: Loading of the DFTB parameters into their associated feed objects
# ----------------------------------------------------------------------
# Construct the `Geometry` and `OrbitalInfo` objects. The former is analogous to the
# ase.Atoms object while the latter provides information about what orbitals
# are present and which atoms they belong two. `OrbitalInfo` is perhaps a poor choice
# of name and `OrbitalInfo` would be more appropriate.
# Construct the Hamiltonian and overlap matrix feeds; but ensure that the DFTB
# parameter set database actually exists first.
if not exists(parameter_db_path):
    raise FileNotFoundError(
        f'The DFTB parameter set database "{parameter_db_path}" could '
        f'not be found, please ensure "example_03_setup.py" has been run.')

# parameter_db_path_std
if not exists(parameter_db_path_std):
    download_dftb_parameter_set(
        "https://github.com/dftbparams/auorg/releases/download/v1.1.0/auorg-1-1.tar.xz",
        parameter_db_path_std)

# Identify which species are present
species = torch.tensor([1, 6, 7, 8])
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Load the Hamiltonian feed model
h_feed = VcrSkFeed.from_database(parameter_db_path, species, 'hamiltonian')
h_feed_std = SkFeed.from_database(
    parameter_db_path_std, species, 'hamiltonian')

# Load the overlap feed model
s_feed = VcrSkFeed.from_database(parameter_db_path, species, 'overlap')
s_feed_std = SkFeed.from_database(
    parameter_db_path_std, species, 'overlap')

# Load the occupation feed object
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)
o_feed_std = SkfOccupationFeed.from_database(parameter_db_path_std, species)

# Load the Hubbard-U feed object
u_feed = HubbardFeed.from_database(parameter_db_path, species)
u_feed_std = HubbardFeed.from_database(parameter_db_path_std, species)

# 2.3: Construct the SCC-DFTB calculator object
# ---------------------------------------------
# As this is a minimal working example, no optional settings are provided to the
# calculator object.
dftb_calculator_init = Dftb2(h_feed, s_feed, o_feed, u_feed)
dftb_calculator_init_std = Dftb2(h_feed_std, s_feed_std, o_feed_std, u_feed_std)

# 2.4: Construct machine learning object
def build_optim(dftb_calculator, dataloder, global_r):
    """Build optimizer for VCR training."""
    # For global compression radii, optimize each atom specie parameters
    comp_r0 = torch.nn.parameter.Parameter(torch.tensor([3.0, 2.7, 2.2, 2.3]))
    # comp_r0.requires_grad_(True)

    ml_onsite, onsite_dict = [], {}
    for key, val in dftb_calculator.h_feed.on_sites.items():
        for l in shell_dict[int(key)]:
            onsite_dict.update({(key, l): val[int(l ** 2)].requires_grad_(True)})
            ml_onsite.append({'params': onsite_dict[(key, l)], 'lr': onsite_lr})

    optimizer = getattr(torch.optim, 'Adam')([{'params': comp_r0, 'lr': lr}])
    return comp_r0, onsite_dict, optimizer


# ================= #
# STEP 3: Execution #
# ================= #
def calculate_losses(calculator: Calculator, data: Any) -> Tensor:
    """An example function computing the loss of the model.

    Args:
        calculator: calculator object via which target properties can be
            calculated.
        targets: target data to which the model should be fitted.

    Returns:
        loss: the computed loss.

    """
    loss = 0.0

    for key in targets:
        key = 'q_final_atomic' if key == 'charge' else key
        loss += criterion(calculator.__getattribute__(key), data.data[key])

    return loss


def single_fit(dftb_calculator, dataloder, n_batch, global_r):
    random.seed = 0
    random_idx = random.sample(torch.arange(len(dataloder)).tolist(), len(dataloder))
    indice = torch.split(torch.tensor(random_idx), n_batch)

    comp_r, onsite_dict, optimizer = build_optim(
        dftb_calculator, dataloder, global_r)

    loss_old = 0
    for epoch in range(number_of_epochs):

        data = dataloder[indice[epoch % len(indice)]]
        orbs = OrbitalInfo(data.geometry.atomic_numbers, shell_dict,
                            shell_resolved=shell_resolved)

        this_cr = torch.ones(data.geometry.atomic_numbers.shape)
        for ii, iatm in enumerate(data.geometry.unique_atomic_numbers()):
            this_cr[iatm == data.geometry.atomic_numbers] = comp_r[ii]
        # print(this_cr)

            # if not shell_resolved:
            #     dftb_calculator.h_feed.on_sites = {
            #         iatm: torch.cat([onsite_dict[(iatm, l)].repeat(2 * l + 1).T
            #                           for l in shell_dict[iatm]], -1)
            #         for iatm in data.geometry.unique_atomic_numbers().tolist()}

        # Perform the forwards operation
        dftb_calculator.h_feed.compression_radii = this_cr
        dftb_calculator.s_feed.compression_radii = this_cr
        dftb_calculator(data.geometry, orbs)
        # print(dftb_calculator.h_feed.compression_radii)

        # Calculate the loss
        loss = calculate_losses(dftb_calculator, data)
        optimizer.zero_grad()
        print(epoch, loss)

        # Invoke the autograd engine

        loss.backward(retain_graph=True)
        optimizer.step()

        if torch.abs(loss_old - loss.detach()).lt(tolerance):
            break
        loss_old = loss.detach().clone()

        this_cr = this_cr.detach().clone()
        min_mask = this_cr[this_cr != 0].lt(1.75)
        max_mask = this_cr[this_cr != 0].gt(9.5)

        # To make sure compression radii inside reasonable range
        if min_mask.any():
            with torch.no_grad():
                comp_r.clamp_(min=2.0)
        if max_mask.any():
            with torch.no_grad():
                comp_r.clamp_(max=9.0)

    # store optimized results to dftb calculator
    dftb_calculator.h_feed.compression_radii = comp_r

    return dftb_calculator

for ii, data_fit in enumerate(dataloder_fit):
    dftb_calculator_fit = single_fit(
        dftb_calculator_init, data_fit, n_fit_batch, global_r=global_r)
