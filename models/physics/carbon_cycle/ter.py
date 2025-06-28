# This module contains the model of terrestrial ecosystem respiration (TER).

# load libraries
import torch

# Net primary productivity (NPP)
def compute_npp(gpp_t: torch.Tensor, cue_t: torch.Tensor):

    """
    This function models net primary productivity at the current time step as a function of GPP and a NN learned parameter cue_t (~CUE).

    Arguments:
    gpp_t: GPP (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    cue_t: A NN learned parameter (-) used to compute NPP (batch_size, 1)

    Returns: npp_t containing net primary productivity (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute net primary productivity
    npp_t = gpp_t * cue_t

    # return net primary productivity
    return npp_t

# autotrophic respiration
def compute_Ra(gpp_t: torch.Tensor, npp_t: torch.Tensor):

    """
    This function computes autotrophic respiration (Ra) as a function of GPP and NPP (all at the current time step).

    Arguments:
    gpp_t: GPP (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    npp_t NPP (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: Ra_t containing autotrophic respiration (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute autotrophic respiration
    Ra_t = gpp_t - npp_t

    # return autotrophic respiration
    return Ra_t

# heterotrophic respiration
def compute_Rh(tair_t_celsius: torch.Tensor, basal_resp_rate_t: torch.Tensor, Q10_value: int, ref_temp: int = 15):

    """
    This function computes heterotrophic respiration using Q10 function as a function of the current air temperature and a NN learned variable basal_resp_rate_t (~Basal respiration).

    Arguments:
    tair_t: Air temperature (C) at the current time step. Torch tensor of shape (batch_size, 1)
    basal_resp_rate_t: A NN learned parameter used to compute heterotrophic respiration. Torch tensor of shape (batch_size, 1)
    Q10_value: NN learned global constant (-)
    ref_temp: Reference temperature (Celsius), usually set to 15 degree C (default)

    Returns: Rh_t containing Hetetrophic respiration (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # define Q10 function
    Q10_function = Q10_value**((tair_t_celsius - ref_temp) / 10)

    # compute heterotrophic respiration
    Rh_t = Q10_function * basal_resp_rate_t

    # return heterotrophic respiration
    return Rh_t

# terrestrial ecosystem respiration
def compute_ter(Ra_t: torch.Tensor, Rh_t: torch.Tensor):

    """
    This function computes Terrestrial ecosystem respiration (TER) by adding autotrophic respiration and hetetrophic respiration.

    Arguments:
    Ra_t: Autotrophic respiration (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    Rh_t: Hetetrophic respiration (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: ter_t containing terrestrial ecosystem respiration (TER) (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute TER
    ter_t = Ra_t + Rh_t

    # return TER
    return ter_t