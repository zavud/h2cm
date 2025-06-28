# This module contains the model of gross primary productivity (GPP).

# load libraries
import torch

# gross primary productivity (GPP)
def compute_gpp(T_t: torch.Tensor, CO2_t: torch.Tensor, wue_t: torch.Tensor, beta_co2: torch.Tensor):

    """
    This function computes GPP at the current time step as a function of transpiration, VPD, CO2 concentration in the atm.,
      and a NN learned variable wue_t, and a learned global constant beta_co2

    Arguments:
    T_t: Transpiration (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    CO2_t: CO2 concentration in the atmosphere (ppm) at the current time step. Torch tensor of shape (batch_size, 1)
    wue: A NN learned variable (gC/kgH20) used to compute GPP. Torch tensor of shape (batch_size, 1)
    beta_co2: A NN learned global constant (ppm-1). Torch tensor of shape (0D)

    Returns: gpp_t containing GPP (gC/m2/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute gross primary productivity
    # gpp_t = (T_t * wue_t) # / torch.maximum((torch.sqrt(vpd_t) * CO2_t), epsilon)
    gpp_t = T_t * wue_t * CO2_t * beta_co2

    # return gross primary productivity
    return gpp_t