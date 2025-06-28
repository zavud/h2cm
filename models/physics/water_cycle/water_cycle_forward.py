# This module contains forward computation of full water cycle

# load libraries
import torch

# load all the water cycle modules
from models.physics.water_cycle import evapotranspiration, gw_storage, runoff, snow, soil_gw_recharge, soil_moisture, tws

# run the forward computation of the full water cycle
def run_water_cycle_forward(water_cycle_input: dict):

    """
    This function runs the full water cycle model in the forward computation. It uses all the lower-level functions in each module of water cycle sub-models.
    This function is intended to be used within a hybrid model's forward computation.

    Arguments:
    - water_cycle_input (dict): A dictionary containing the input parameters for the water cycle model. This includes:
      - tair: Air temperature in Kelvin.
      - rainfall: Rainfall in mm/day.
      - snowfall: Snowfall in mm/day.
      - rn: Net radiation in Watts/m^2.
      - alpha_snow_melt, alpha_Es, alpha_T, alpha_r_soil, alpha_r_gw: Temporal NN predictions for various fluxes.
      - fAPAR_nn: predicted Fraction of Absorbed Photosynthetically Active Radiation (-).
      - swe: Snow Water Equivalent in mm.
      - SM: Soil Moisture in mm.
      - GW: Groundwater storage in mm.
      - SM_max_nn: Maximum soil moisture from static NN predictions (mm).
      - alpha_Ei: Static NN prediction for interception evaporation.
      - beta_baseflow: Global constant baseflow processes.

    Returns:
    - A dictionary containing the computed water cycle parameters, including:
      - snow_melt: Snow melt in mm/day.
      - Ei: Interception evaporation in mm/day.
      - Es: Soil evaporation in mm/day.
      - T: Transpiration in mm/day.
      - ET: Evapotranspiration in mm/day.
      - r_soil_fraction: Soil recharge fraction (-).
      - r_soil: Soil recharge in mm/day.
      - r_gw_fraction: Groundwater recharge fraction (-).
      - r_gw: Groundwater recharge in mm/day.
      - runoff_surface_fraction: Surface runoff fraction (-).
      - runoff_surface: Surface runoff in mm/day.
      - baseflow: Baseflow in mm/day.
      - runoff_total: Total runoff in mm/day.
      - tws: Terrestrial water storage in mm.
    """

    ### Extract the input parameters

    # forcing parameters
    tair_t = water_cycle_input["tair_t"]
    prec_t = water_cycle_input["prec_t"]
    rn_t = water_cycle_input["rn_t"]

    # physical states at the previous time step
    swe_t_prev = water_cycle_input["swe_t_prev"]
    SM_t_prev = water_cycle_input["SM_t_prev"]
    GW_t_prev = water_cycle_input["GW_t_prev"]

    # temporal NN predictions
    alpha_snow_melt_t = water_cycle_input["alpha_snow_melt_t"]
    alpha_Es_t = water_cycle_input["alpha_Es_t"]
    alpha_T_t = water_cycle_input["alpha_T_t"]
    alpha_r_soil_t = water_cycle_input["alpha_r_soil_t"]
    alpha_r_gw_t = water_cycle_input["alpha_r_gw_t"]
    fAPAR_nn_t = water_cycle_input["fAPAR_nn_t"]

    # static NN predictions
    SM_max_nn = water_cycle_input["SM_max_nn"]
    alpha_Ei = water_cycle_input["alpha_Ei"]
    # alpha_baseflow = water_cycle_input["alpha_baseflow"]

    # global constants
    beta_snow = water_cycle_input["beta_snow"]
    beta_baseflow = water_cycle_input["beta_baseflow"]

    ### Snow model ###

    # convert the current air temperature from kelvin to C
    tair_t_celsius = snow.convert_tair(tair_t = tair_t)

    # snow accumulation
    snow_acc_t = snow.compute_snow_acc(prec_t = prec_t, beta_snow = beta_snow, tair_t_celsius = tair_t_celsius)

    # snow melt
    snow_melt_t = snow.compute_snow_melt(swe_t_prev = swe_t_prev, tair_t_celsius = tair_t_celsius, alpha_snow_melt_t = alpha_snow_melt_t)

    # update snow water equivalent (swe)
    swe_t = snow.update_swe(swe_t_prev = swe_t_prev, snow_acc_t = snow_acc_t, snow_melt_t = snow_melt_t)

    ### Evapotranspiration ###

    # convert rn from (Watts / m^2) to mm
    rn_t_mm = evapotranspiration.convert_rn_to_mm(rn_t = rn_t)

    # convert negative values of rn_t_mm to 0 if they are negative
    rn_t_mm = evapotranspiration.make_zero_if_negative(rn_t_mm = rn_t_mm)

    # rainfall
    rainfall_t = evapotranspiration.compute_rainfall(prec_t = prec_t, tair_t_celsius = tair_t_celsius)

    # interception evaporation
    Ei_t = evapotranspiration.compute_Ei(rainfall_t = rainfall_t, fAPAR_nn_t = fAPAR_nn_t, rn_t_mm = rn_t_mm, alpha_Ei_t = alpha_Ei)

    # update net radiation
    rn_t_mm_remaining = evapotranspiration.update_Rn(rn_t_mm = rn_t_mm, water_flux_t = Ei_t)

    # potential evapotranspiration
    ET_pot_t = evapotranspiration.compute_ET_pot(rn_t_mm = rn_t_mm_remaining, SM_t = SM_t_prev)

    # soil evaporation
    Es_t = evapotranspiration.compute_Es3(fAPAR_nn_t = fAPAR_nn_t, ET_pot_t = ET_pot_t, alpha_Es_t = alpha_Es_t)

    # update net radiation
    rn_t_mm_remaining = evapotranspiration.update_Rn(rn_t_mm = rn_t_mm_remaining, water_flux_t = Es_t)

    # update soil moisture as a function of SM_t and Es_t
    SM_t = soil_moisture.update_SM(SM_current_or_prev = SM_t_prev, r_soil_t = torch.tensor(0.0), T_t = torch.tensor(0.0), Es_t = Es_t)
    
    # update potential evapotranspiration
    ET_pot_t = evapotranspiration.compute_ET_pot(rn_t_mm = rn_t_mm_remaining, SM_t = SM_t)

    # transpiration
    T_t = evapotranspiration.compute_T2(fAPAR_nn_t = fAPAR_nn_t, ET_pot_t = ET_pot_t, alpha_T_t = alpha_T_t)

    # update net radiation
    rn_t_mm_remaining = evapotranspiration.update_Rn(rn_t_mm = rn_t_mm_remaining, water_flux_t = T_t)

    # update soil moisture as a function of SM_t_Es and T_t
    SM_t = soil_moisture.update_SM(SM_current_or_prev = SM_t, r_soil_t = torch.tensor(0.0), T_t = T_t, Es_t = torch.tensor(0.0))

    # evapotranspiration
    ET_t = evapotranspiration.compute_ET(Ei_t = Ei_t, Es_t = Es_t, T_t = T_t)

    ### Soil & Groundwater recharge ###

    # water input
    water_input_t = soil_gw_recharge.compute_water_input(rainfall_t = rainfall_t, snow_melt_t = snow_melt_t, Ei_t = Ei_t)

    # soil recharge fraction
    r_soil_t_fraction = soil_gw_recharge.compute_r_soil_fraction2(SM_t = SM_t, SM_max_nn = SM_max_nn, water_input_t = water_input_t, alpha_r_soil_t = alpha_r_soil_t)

    # soil recharge & remaining water that cannot enter the soil
    r_soil_t = soil_gw_recharge.compute_r_soil2(water_input_t = water_input_t, r_soil_t_fraction = r_soil_t_fraction)
    
    # update soil moisture as a function of SM_t_T and r_soil_t
    SM_t = soil_moisture.update_SM(SM_current_or_prev = SM_t, r_soil_t = r_soil_t, T_t = torch.tensor(0.0), Es_t = torch.tensor(0.0))

    # compute relative soil moisture
    rel_SM_t = soil_moisture.compute_relative_SM(SM_t = SM_t, SM_max_nn = SM_max_nn)

    # groundwater recharge
    r_gw_fraction_t = soil_gw_recharge.compute_r_gw_frac(r_soil_t_fraction = r_soil_t_fraction, alpha_r_gw_t = alpha_r_gw_t)
    r_gw_t = soil_gw_recharge.compute_r_gw2(water_input_t = water_input_t, r_gw_t_fraction_t = r_gw_fraction_t)

    ### Runoff ###

    # surface runoff
    runoff_surface_fraction_t = runoff.runoff_surface_frac(r_soil_t_fraction = r_soil_t_fraction, alpha_r_gw_t = alpha_r_gw_t)
    runoff_surface_t = runoff.compute_runoff_surface2(water_input_t = water_input_t, runoff_surface_fraction_t = runoff_surface_fraction_t)

    # baseflow
    baseflow_t = runoff.compute_baseflow(GW_t_prev = GW_t_prev, beta_baseflow = beta_baseflow)

    # total runoff
    runoff_total_t = runoff.compute_runoff_total(runoff_surface_t = runoff_surface_t, baseflow_t = baseflow_t)


    ### Groundwater storage ###

    # groundwater storage
    GW_t = gw_storage.update_GW(GW_t_prev = GW_t_prev, r_gw_t = r_gw_t, baseflow_t = baseflow_t)

    ### Terrestrial water storage (TWS) ###
    tws_t = tws.compute_tws(swe_t = swe_t, GW_t = GW_t, SM_t = SM_t)

    # store all the needed parameters after the full forward run of water cycle in a dictionary
    water_cycle_output = {
         "snow_acc_t": snow_acc_t,
         "snow_melt_t": snow_melt_t,
         "swe_t": swe_t,
         "Ei_t": Ei_t,
         "Es_t": Es_t,
         "T_t": T_t,
         "ET_t": ET_t,
         "water_input_t": water_input_t,
         "r_soil_t_fraction": r_soil_t_fraction,
         "r_soil_t": r_soil_t,
         "r_gw_fraction_t": r_gw_fraction_t,
         "r_gw_t": r_gw_t,
         "SM_t": SM_t,
         "rel_SM_t": rel_SM_t,
         "runoff_surface_fraction_t": runoff_surface_fraction_t,
         "runoff_surface_t": runoff_surface_t,
         "baseflow_t": baseflow_t,
         "runoff_total_t": runoff_total_t,
         "GW_t": GW_t,
         "tws_t": tws_t,
         "prec_actual": snow_acc_t + rainfall_t,
         "tair_t_celsius": tair_t_celsius
    }

    # return all the computed parameters
    return water_cycle_output