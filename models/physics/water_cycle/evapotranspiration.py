# This module contains the snow model of the coupled water-carbon cycles

# load libraries
import torch

# Rainfall
def compute_rainfall(prec_t: torch.Tensor, tair_t_celsius: torch.Tensor):

    """
    This function computes rainfall as a function of precipitation and air temperature. Rainfall is simply the precipitation, if the air temperature is greater than 0. 
     Otherwise, it is equal to 0.

    Arguments:
    prec_t: Precipitation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    tair_t_celsius: Air temperature (Celsius) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: rainfall_t containing the amount of rainfall (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # check whether the types of the given arguments are correct
    # assert isinstance(prec_t, torch.Tensor), "prec_t must be a type of torch.Tensor"
    # assert isinstance(tair_t_celsius, torch.Tensor), "tair_t must be a type of torch.Tensor"

    # find whether the weather condition for the rain is met (air temp > 0 Celsius)
    # days where air temp is greater than 0 will have a value of 1, otherwise 0
    is_raining = torch.greater(tair_t_celsius, torch.tensor(0)) * torch.tensor(1.0) # of shape (batch_size, 1)

    # compute rainfall - which is simply precipitation if is_raining=1, otherwise 0
    rainfall_t = prec_t * is_raining # of shape (batch_size, 1)

    # return the computed rainfall
    return rainfall_t

# interception evaporation
def compute_Ei(rainfall_t: torch.Tensor, fAPAR_nn_t: torch.Tensor, rn_t_mm: torch.Tensor, alpha_Ei_t: torch.Tensor):

    """
    This function computes interception evaporation as a function of rainfall, fAPAR and a NN learned parameter alpha_Ei.

    Arguments:
    rainfall_t: The amount of rainfall (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    fAPAR_nn_t: The fraction of photosynthetically active radiation (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    alpha_Ei_t: A NN learned parameter used for modelling interception evaporation. Torch tensor of shape (batch_size, 1)

    Returns: Ei_t containing interception evaporation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute ~maximum water holding capacity (max_whc_t) of plants
    max_whc_t = fAPAR_nn_t * alpha_Ei_t # of shape (batch_size, 1)

    # compute potential interception evaporation (Ei) as a minimum of rainfall and max_whc of plants
    Ei_pot_t = torch.minimum(rainfall_t, max_whc_t) # of shape (batch_size, 1)

    # compute interception evaporation (Ei) as a minimum of potential Ei and the available energy
    Ei_t = torch.minimum(Ei_pot_t, rn_t_mm)

    # return interception evaporation
    return Ei_t

# convert rn from Watts per square meter per day to mm per day
def convert_rn_to_mm(rn_t: torch.Tensor):

    """
    This function converts Net radiation's unit from Watts per square meter per day to mm per day.

    Please see: https://www.researchgate.net/post/How_to_convert_30minute_evapotranspiration_in_watts_to_millimeters

    Arguments:
    rn_t: Net radiation (Watts / m^2) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: rn_t_mm containing the Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # convert net radiation from (Watts / m^2) to (MJ / m^2)
    rn_t_MJ = rn_t * 0.0864

    # convert net radiation's unit (rn_t) from (MJ / m^2) to mm (depth).
    rn_t_mm = rn_t_MJ / 2.45 # of shape (batch_size, 1)

    # return net radiation with a unit of mm
    return rn_t_mm

# convert negative values of rn_t_mm to 0
def make_zero_if_negative(rn_t_mm: torch.Tensor):

    """
    This function converts negative values of net radiation (mm) to 0 for physical computation (not used as an input).

    Arguments:
    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: rn_t_mm_converted containing the Net radiation (mm/day) negative values converted to 0s at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # convert the values of rn_t_mm to 0 if they are negative
    rn_t_mm_converted = torch.maximum(rn_t_mm, torch.tensor(0.0))

    # return the converted tensor
    return rn_t_mm_converted

# update net radiation
def update_Rn(rn_t_mm: torch.Tensor, water_flux_t: torch.Tensor):

    """
    This function subtracts interception evaporation (Ei) or soil evaporation (Es) from net radiation (Rn).

    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    water_flux_t: Interception evaporation or soil evaporation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns:
    rn_t_mm: Updated net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # update net radiation by subtracting Ei from it
    rn_t_mm = rn_t_mm - water_flux_t

    # return the updated net radiation
    return rn_t_mm


# compute potential evapotranspiration
def compute_ET_pot(rn_t_mm: torch.Tensor, SM_t: torch.Tensor):

    """
    This function computes potential evapotranspiration as a function of net radiation and soil moisture.

    Arguments:
    rn_t_mm: Net radiation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    SM_t: Soil moisture (mm) at the current time step. Torch tensor of shape (batch_size, 1)
    
    Returns: ET_pot_t containing potential evapotranspiration (mm/day). Torch tensor of shape (batch_size, 1)
    """

    # compute potential evapotranspiration as a minimum of net radiation (mm) and soil moisture
    ET_pot_t = torch.minimum(rn_t_mm, SM_t)

    # return the potential evapotranspiration
    return ET_pot_t

# soil evaporation 3 (new implementation)
def compute_Es3(fAPAR_nn_t: torch.Tensor, ET_pot_t: torch.tensor, alpha_Es_t: torch.tensor):

    """
    This function computes soil evaporation as a function of fapar, potential ET and two NN learned parameters.

    Arguments:
    fAPAR_nn_t: The fraction of photosynthetically active radiation (-) (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    ET_pot_t:  Potential evapotranspiration (mm/day). Torch tensor of shape (batch_size, 1)
    alpha_Es_t: NN learned parameter varied both in space and time. Torch tensor of shape (batch_size, 1)
    
    Returns: Es_t containing soil evaporation (mm/day). Torch tensor of shape (batch_size, 1)
    """

    # compute/model soil coverage as approx. being (1 - vegetation) in the grid
    soil_coverage_t = 1 - fAPAR_nn_t # of shape (batch_size, 1)

    # compute potential soil evaporation
    Es_pot_t = soil_coverage_t * ET_pot_t

    # compute final soil evaporation using NN learned parameters
    Es_t = Es_pot_t * alpha_Es_t

    # return the final soil evaporation
    return Es_t

# transpiration 2
def compute_T2(fAPAR_nn_t: torch.Tensor, ET_pot_t: torch.Tensor, alpha_T_t: torch.Tensor):

    """
    This function computes transpiration as a function of fapar, potential ET and two NN learned parameters.

    Arguments:
    fAPAR_nn_t: The fraction of photosynthetically active radiation (-) (400-700 nm) absorbed by green vegetation (predicted) at the current time step. Torch tensor of shape (batch_size, 1)
    ET_pot_t:  Potential evapotranspiration (mm/day). Torch tensor of shape (batch_size, 1)
    alpha_T_t: NN learned parameter varied both in space and time. Torch tensor of shape (batch_size, 1)
    
    Returns: Es_t containing soil evaporation. Torch tensor of shape (batch_size, 1)
    """

    # compute/model plant coverage as approx. being vegetation/fapar in the grid
    plant_coverage_t = fAPAR_nn_t # of shape (batch_size, 1)

    # compute potential soil evaporation
    T_pot_t = plant_coverage_t * ET_pot_t

    # compute final soil evaporation using NN learned parameters
    T_t = T_pot_t * alpha_T_t

    # return the final soil evaporation
    return T_t

# evapotranspiration
def compute_ET(Ei_t: torch.Tensor, Es_t: torch.Tensor, T_t: torch.Tensor):

    """
    This function computes evapotranspiration as a sum of interception evaporation, soil evaporation and transpiration at the
     current time step.

    Arguments:
    Ei_t: Interception evaporation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    Es_t: Soil evaporation (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    T_t: Transpiration (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)

    Returns: ET_t containing evapotranspiration (mm/day) at the current time step. Torch tensor of shape (batch_size, 1)
    """

    # compute evapotranspiration
    ET_t = Ei_t + Es_t + T_t # of shape (batch_size, 1)

    # return evapotranspiration
    return ET_t