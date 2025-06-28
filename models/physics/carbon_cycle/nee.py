# This module contains the model of Net ecosystem exchange (NEE)

# load libraries
import torch

# net ecosystem productivity (NEP)
def compute_nep(gpp_t: torch.Tensor, ter_t: torch.Tensor):

    """
    This function computes net ecosystem productivity (NEP) as a difference between gross primary productivity (GPP) and terrestrial ecosystem respiration (TER).

    Arguments:
    gpp_t: GPP (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    ter_t: TER (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: nep_t containing net ecosystem productivity (NEP) (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute NEP
    nep_t = gpp_t - ter_t

    # return NEP
    return nep_t

    # Net ecosystem exchange
def compute_nee(nep_t: torch.Tensor):

    """
    This function takes NEP at the current time step and negates its sign, which is net ecosystem exchange (NEE).

    Arguments:
    nep_t: Net ecosystem productivity (NEP) (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: nee_t containing net ecosystem exchange (NEE) (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute NEE
    nee_t = -nep_t

    # return NEE
    return nee_t