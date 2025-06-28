# This module contains forward computation of full carbon cycle

# import carbon cycle modules
from models.physics.carbon_cycle import gpp, ter, nee

# run the forward computation of the full carbon cycle
def run_carbon_cycle_forward(carbon_cycle_input: dict):

    """
    This function runs the full carbon cycle model in the forward computation. It uses all the lower-level functions in each module of carbon cycle sub-models.
    This function is intended to be used within the hybrid model's forward computation.

    Arguments:
    - carbon_cycle_input (dict): A dictionary containing the inputs to the carbon cycle, including:
      - tair_celsius: Air temperature in Celsius.
      - CO2: Carbon dioxide concentration in ppm.
      - wue: Water Use Efficiency from temporal NN predictions in gC/kgH2O.
      - cue: Carbon Use Efficiency from temporal NN predictions (-).
      - basal_resp_rate: Basal respiration rate from temporal NN predictions (gC/m2/day).
      - beta_q10, beta_co2: Global constants for Q10 and CO2 processes.
      - T: Transpiration in mm/day.

    Returns:
    - A dictionary containing the computed carbon cycle parameters (gC/m2/day), including:
      - gpp: Gross Primary Production.
      - npp: Net Primary Production.
      - Ra: Autotrophic Respiration.
      - Rh: Heterotrophic Respiration.
      - ter: Total Ecosystem Respiration.
      - nep: Net Ecosystem Production.
      - nee: Net Ecosystem Exchange.
    """

    ### Extract the input parameters

    # forcing parameters
    tair_t_celsius = carbon_cycle_input["tair_t_celsius"]
    CO2_t = carbon_cycle_input["CO2_t"]

    # temporal NN predictions
    wue_t = carbon_cycle_input["wue_t"]
    cue_t = carbon_cycle_input["cue_t"]
    basal_resp_rate_t = carbon_cycle_input["basal_resp_rate_t"]

    # global constants
    beta_q10 = carbon_cycle_input["beta_q10"]
    beta_co2 = carbon_cycle_input["beta_co2"]

    # intermediate predictions from water cycle
    T_t = carbon_cycle_input["T_t"]

    ### GPP ###

    # compute GPP
    gpp_t = gpp.compute_gpp(T_t = T_t, CO2_t = CO2_t, wue_t = wue_t, beta_co2 = beta_co2)

    ### TER ###

    # compute NPP
    npp_t = ter.compute_npp(gpp_t = gpp_t, cue_t = cue_t)

    # compute Ra
    Ra_t = ter.compute_Ra(gpp_t = gpp_t, npp_t = npp_t)

    # compute Rh
    Rh_t = ter.compute_Rh(tair_t_celsius = tair_t_celsius, basal_resp_rate_t = basal_resp_rate_t, Q10_value = beta_q10, ref_temp = 15)

    # compute TER
    ter_t = ter.compute_ter(Ra_t = Ra_t, Rh_t = Rh_t)

    ### NEE ###

    # compute NEP
    nep_t = nee.compute_nep(gpp_t = gpp_t, ter_t = ter_t)

    # compute NEE
    nee_t = nee.compute_nee(nep_t = nep_t)

    # store all outputs of carbon cycle
    carbon_cycle_output = {
        "gpp_t": gpp_t,
        "npp_t": npp_t,
        "Ra_t": Ra_t,
        "Rh_t": Rh_t,
        "ter_t": ter_t,
        "nep_t": nep_t,
        "nee_t": nee_t
    }

    # return carbon cycle outputs
    return carbon_cycle_output