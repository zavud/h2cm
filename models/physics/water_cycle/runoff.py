# This module contains the runoff model.

# load libraries
import torch

# surface runoff fraction
def runoff_surface_frac(r_soil_t_fraction: torch.Tensor, alpha_r_gw_t: torch.Tensor):

    """
    This function computes surface runoff fraction.
    
    Arguments:
    r_soil_t_fraction: Soil recharge fraction (-) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_r_gw_t: NN learned parameter used to model groundwater recharge. It is between 0 and 1. Torch tensor of shape (batch_size, 1)

    Returns:
    runoff_surface_fraction_t: Surface runoff fraction (-) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute surface runoff (fraction)
    runoff_surface_fraction_t = (1 - r_soil_t_fraction) * (1 - alpha_r_gw_t)

    # return the surface runoff fraction
    return runoff_surface_fraction_t

# surface runoff
def compute_runoff_surface2(water_input_t: torch.Tensor, runoff_surface_fraction_t: torch.Tensor):

    """
    This function computes surface runoff at the current time step.

    Arguments:
    water_input_t: Water input (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    runoff_surface_fraction_t: Surface runoff fraction (-) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: runoff_surface_t containing surface runoff (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute surface runoff
    runoff_surface_t = runoff_surface_fraction_t * water_input_t

    # return surface runoff
    return runoff_surface_t

# baseflow
def compute_baseflow(GW_t_prev: torch.Tensor, beta_baseflow: torch.Tensor):

    """
    This function computes baseflow as a product between groundwater storage at the previous time step and a NN learned scaler beta_baseflow.

    Arguments:
    GW_t_prev: Groundwater storage (mm) at the previous time step. Torch tensor of shape (batch_size, 1)
    beta_baseflow: A NN learned static parameter (fixed in time and but varied in space). Torch tensor of shape (batch_size, 1)

    Returns: baseflow_t containing baseflow (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute baseflow
    baseflow_t = GW_t_prev * beta_baseflow

    # return baseflow
    return baseflow_t

# total runoff
def compute_runoff_total(runoff_surface_t: torch.Tensor, baseflow_t: torch.Tensor):

    """
    This function computes the total runoff as a sum of surface runoff and baseflow at the current time step.

    Arguments:
    runoff_surface_t: Surface runoff (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    baseflow_t: Baseflow (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: runoff_total_t containing total runoff (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute total runoff
    runoff_total_t = runoff_surface_t + baseflow_t

    # return total runoff
    return runoff_total_t