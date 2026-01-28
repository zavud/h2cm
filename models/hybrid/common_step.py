# this module contains the common step that can be used for training/validation/testing.

# load libraries
import torch

# load custom modules
from models.hybrid import hybrid_helpers as hh
from models.physics.water_cycle import tws

def common_step(self, batch):

    """
    This function implements the training step of the hybrid ML model of full coupled water-carbon cycle. This function is meant to be used within the validation/test_step of pytorch lightning.

    Arguments:
    self: A pytorch lightning model where the full forward run of the hybrid model of water cycle has been implemented.
    batch: Parameter required by pytorch lightning's training/validation/test_step

    Returns: loss_sum containing the final summed loss of all the predictions
    """

    # get the features and constraints from the batch
    forcing, static, constraints_static, constraints, grid_area, _ = batch

    # store the number of examples/grids in the current batch (needed for initialising lstm's hidden and cell states)
    batch_size = forcing.shape[0]

    # initialise the physical states with zeros
    swe0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)
    SM0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)
    GW0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)
    fAPAR_nn_0 = torch.zeros(size = (batch_size, 1), dtype = torch.float32)

    # initialise the hidden and cell states with 0's
    h0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    c0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)

    # initialise npp flux with ~1.17 which is the global value from TRENDY v11 (median across members)
    npp0 = torch.full(size = (batch_size, 1), fill_value = 1.1738447874691162)

    # initialise the hidden and cell states with 0's
    h0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    c0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    h_win_partition_smelt0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    c_win_partition_smelt0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    h_cue_fapar0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    c_cue_fapar0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    h_rb_alpha_Es0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)
    c_rb_alpha_Es0 = torch.zeros(size = (batch_size, self.hidden_size_lstm), dtype = torch.float32)

    # store the state variables in a dictionary
    states0 = {
        "swe_t": swe0,
        "SM_t": SM0,
        "GW_t": GW0,
        "fAPAR_nn_t": fAPAR_nn_0,
        "h_t": h0,
        "c_t": c0,
        "npp_t": npp0,
        "h_t_win_partition_smelt": h_win_partition_smelt0,
        "c_t_win_partition_smelt": c_win_partition_smelt0,
        "h_t_cue_fapar": h_cue_fapar0,
        "c_t_cue_fapar": c_cue_fapar0,
        "h_t_rb_alpha_Es": h_rb_alpha_Es0,
        "c_t_rb_alpha_Es": c_rb_alpha_Es0 
    }

    # run forward computation on spin-up mode
    with torch.no_grad(): # detach from the computational graph
        _, _, _, states_spinup = self(forcing = forcing, static = static, states_initial = states0)

    # get static constraints
    cue_ma_values = constraints_static["cue"] # cue mean annual

    # get values & dates from the constraints
    tws_values, tws_dates = constraints["tws"]
    et_values, et_dates = constraints["et_xbase"]
    q_values, q_dates = constraints["q_v2"]
    swe_values, swe_dates = constraints["swe_bc"]
    fapar_values, fapar_dates = constraints["fapar"]
    gpp_values, gpp_dates = constraints["gpp"]
    nee_values, nee_dates = constraints["nee"]
    nee_iav_values, nee_iav_dates = constraints["nee_iav"]

    # add a new dimension to make the dimension equivalent to predictions' dimensions
    tws_values = tws_values.unsqueeze(2)
    et_values = et_values.unsqueeze(2)
    q_values = q_values.unsqueeze(2)
    swe_values = swe_values.unsqueeze(2)
    fapar_values = fapar_values.unsqueeze(2)
    gpp_values = gpp_values.unsqueeze(2)
    nee_values = nee_values.unsqueeze(2)
    nee_iav_values = nee_iav_values.unsqueeze(2)
    cue_ma_values = cue_ma_values.unsqueeze(1)

    # make dates of constraints a long 1D tensor
    tws_dates = tws_dates[0, :]
    et_dates = et_dates[0, :]
    q_dates = q_dates[0, :]
    swe_dates = swe_dates[0, :]
    fapar_dates = fapar_dates[0, :]
    gpp_dates = gpp_dates[0, :]
    nee_dates = nee_dates[0, :]
    nee_iav_dates = nee_iav_dates[0, :]

    # run forward hybrid computation
    preds_nn, preds_hybrid, learned_constants, _ = self(forcing = forcing, static = static, states_initial = states_spinup)

    # get predicted constraint variables from the dictionary. All -> of shape (batch_size, num_time_steps, 1)
    tws_predicted = preds_hybrid["tws_ts"]
    et_predicted = preds_hybrid["ET_ts"]
    q_predicted = preds_hybrid["runoff_total_ts"]
    swe_predicted = preds_hybrid["swe_ts"]
    fapar_predicted = preds_nn["fAPAR_nn_ts"]
    gpp_predicted = preds_hybrid["gpp_ts"]
    nee_predicted = preds_hybrid["nee_ts"]
    npp_predicted = preds_hybrid["npp_ts"] # npp is needed to compute predicted cue_ma

    # compute mean annual gpp and npp (batch_size, 1)
    gpp_predicted_ma = gpp_predicted.mean(dim = 1)
    npp_predicted_ma = npp_predicted.mean(dim = 1)
    # compute predicted cue_ma
    cue_ma_predicted = torch.where(gpp_predicted_ma < 0.1, torch.tensor(0.5), npp_predicted_ma / gpp_predicted_ma) # we need to make sure we do not end up doing npp/(gpp=0)
    # standardize cue_ma_predicted
    cue_ma_predicted_standardized = hh.standardize_single(data = cue_ma_predicted, mean = self.mean_cue, std = self.std_cue)

    # compute interranual variability (iav) of predicted nee
    nee_predicted4iav, nee_predicted4iav_dates_filtered = hh.filter_matching_dates(data_pred = nee_predicted, dates_pred = self.dates_prediction, dates_target = nee_iav_dates)
    nee_predicted4iav_monthly  = hh.aggregate_monthly_nan(nee_predicted4iav, nee_predicted4iav_dates_filtered) # aggregate the data to monthly
    nee_msc_predicted = nee_predicted4iav_monthly.view(batch_size, int(nee_predicted4iav_monthly.shape[1] / 12), 12).mean(dim = 1).unsqueeze(2) # msc
    nee_msc_predicted =  nee_msc_predicted.repeat(1, int(nee_predicted4iav_monthly.shape[1]  / 12), 1) # repeat the second dimension by num. of years times (for broadcasting)
    # nee_iav_predicted: shape (batch_size, num_time_steps, 1)
    nee_iav_predicted = nee_predicted4iav_monthly - nee_msc_predicted # we define iav/anomaly as difference between predictions and the msc

    # filter the dates and values of predictions to match the dates of the corresponding target
    tws_predicted, tws_predicted_dates_filtered = hh.filter_matching_dates(data_pred = tws_predicted, dates_pred = self.dates_prediction, dates_target = tws_dates)
    et_predicted, et_predicted_dates_filtered = hh.filter_matching_dates(data_pred = et_predicted, dates_pred = self.dates_prediction, dates_target = et_dates)
    q_predicted, q_predicted_dates_filtered = hh.filter_matching_dates(data_pred = q_predicted, dates_pred = self.dates_prediction, dates_target = q_dates)
    swe_predicted, swe_predicted_dates_filtered = hh.filter_matching_dates(data_pred = swe_predicted, dates_pred = self.dates_prediction, dates_target = swe_dates)
    fapar_predicted, fapar_predicted_dates_filtered = hh.filter_matching_dates(data_pred = fapar_predicted, dates_pred = self.dates_prediction, dates_target = fapar_dates)
    gpp_predicted, gpp_predicted_dates_filtered = hh.filter_matching_dates(data_pred = gpp_predicted, dates_pred = self.dates_prediction, dates_target = gpp_dates)
    nee_predicted, nee_predicted_dates_filtered = hh.filter_matching_dates(data_pred = nee_predicted, dates_pred = self.dates_prediction, dates_target = nee_dates)
    # nee_iav_predicted_standardized is filtered above

    # aggregate the needed targets to monthly
    tws_target_monthly = hh.aggregate_monthly_nan(tws_values, tws_dates)
    et_target_monthly = hh.aggregate_monthly_nan(et_values, et_dates)
    q_target_monthly = hh.aggregate_monthly_nan(q_values, q_dates)
    fapar_target_monthly = hh.aggregate_monthly_nan(fapar_values, fapar_dates)
    swe_target_monthly = hh.aggregate_monthly_nan(swe_values, swe_dates)
    gpp_target_monthly = hh.aggregate_monthly_nan(gpp_values, gpp_dates)
    nee_target_monthly = hh.aggregate_monthly_nan(nee_values, nee_dates)
    nee_iav_target_monthly = hh.aggregate_monthly_nan(nee_iav_values, nee_iav_dates)

    # agregate the needed predictions to monthly
    tws_predicted_monthly = hh.aggregate_monthly_nan(tws_predicted, tws_predicted_dates_filtered)
    et_predicted_monthly = hh.aggregate_monthly_nan(et_predicted, et_predicted_dates_filtered)
    q_predicted_monthly = hh.aggregate_monthly_nan(q_predicted, q_predicted_dates_filtered)
    fapar_predicted_monthly = hh.aggregate_monthly_nan(fapar_predicted, fapar_predicted_dates_filtered)
    swe_predicted_monthly = hh.aggregate_monthly_nan(swe_predicted, swe_predicted_dates_filtered)
    gpp_predicted_monthly = hh.aggregate_monthly_nan(gpp_predicted, gpp_predicted_dates_filtered)
    nee_predicted_monthly = hh.aggregate_monthly_nan(nee_predicted, nee_predicted_dates_filtered)
    # nee_iav_predicted was aggregated to monthly above

    # compute mean seasonal cycles of gpp, et & runoff
    ## gpp
    gpp_target_monthly = hh.compute_msc_monthly(gpp_target_monthly)
    gpp_predicted_monthly = hh.compute_msc_monthly(gpp_predicted_monthly)
    ## et
    et_target_monthly = hh.compute_msc_monthly(et_target_monthly)
    et_predicted_monthly = hh.compute_msc_monthly(et_predicted_monthly)
    # runoff
    q_target_monthly = hh.compute_msc_monthly(q_target_monthly)
    q_predicted_monthly = hh.compute_msc_monthly(q_predicted_monthly)

    # compute batch level (~global) iav
    # shape: (1, num_time_steps, 1)
    nee_iav_target_avg = hh.weighted_mean(nee_iav_target_monthly, weights = grid_area)
    nee_iav_predicted_avg = hh.weighted_mean(nee_iav_predicted, weights = grid_area)

    # temporally smooth the time series for NEE IAV using a window size of 7
    nee_iav_target_avg = hh.temporal_smooth(nee_iav_target_avg, window_size = 7)
    nee_iav_predicted_avg = hh.temporal_smooth(nee_iav_predicted_avg, window_size = 7)

    # center target tws around 0 to make it comparable to predicted tws anomalies
    tws_target_monthly = tws.compute_tws_anomaly(tws_target_monthly)
    tws_predicted_monthly = tws.compute_tws_anomaly(tws_predicted_monthly)

    # standardise the constraints
    tws_target_monthly = hh.standardize_single(data = tws_target_monthly, mean = torch.tensor(0.0), std = self.std_tws)
    et_target_monthly = hh.standardize_single(data = et_target_monthly, mean = self.mean_et, std = self.std_et)
    q_target_monthly = hh.standardize_single(data = q_target_monthly, mean = self.mean_q, std = self.std_q)
    swe_target_monthly = hh.standardize_single(data = swe_target_monthly, mean = self.mean_swe, std = self.std_swe)
    fapar_target_monthly = hh.standardize_single(data = fapar_target_monthly, mean = self.mean_fapar, std = self.std_fapar)
    gpp_target_monthly = hh.standardize_single(data = gpp_target_monthly, mean = self.mean_gpp, std = self.std_gpp)
    nee_target_monthly = hh.standardize_single(data = nee_target_monthly, mean = self.mean_nee, std = self.std_nee)
    nee_iav_target_avg = hh.standardize_single(data = nee_iav_target_avg, mean = self.mean_nee_iav, std = self.std_nee_iav)
    cue_ma_values_standardized = hh.standardize_single(data = cue_ma_values, mean = self.mean_cue, std = self.std_cue)

    # standardize the predictions
    tws_predicted_monthly = hh.standardize_single(data = tws_predicted_monthly, mean = torch.tensor(0.0), std = self.std_tws)
    et_predicted_monthly = hh.standardize_single(data = et_predicted_monthly, mean = self.mean_et, std = self.std_et)
    q_predicted_monthly = hh.standardize_single(data = q_predicted_monthly, mean = self.mean_q, std = self.std_q)
    swe_predicted_monthly = hh.standardize_single(data = swe_predicted_monthly, mean = self.mean_swe, std = self.std_swe)
    fapar_predicted_monthly = hh.standardize_single(data = fapar_predicted_monthly, mean = self.mean_fapar, std = self.std_fapar)
    # baseflow_k_predicted_standardized = hh.standardize_single(data = baseflow_k_predicted, mean = self.mean_baseflow_k, std = self.std_baseflow_k)
    gpp_predicted_monthly = hh.standardize_single(data = gpp_predicted_monthly, mean = self.mean_gpp, std = self.std_gpp)
    nee_predicted_monthly = hh.standardize_single(data = nee_predicted_monthly, mean = self.mean_nee, std = self.std_nee)
    nee_iav_predicted_avg = hh.standardize_single(data = nee_iav_predicted_avg, mean = self.mean_nee_iav, std = self.std_nee_iav)
    
    # compute the loss for the predictions
    loss_tws_monthly = hh.compute_nan_mse_time(pred = tws_predicted_monthly, target = tws_target_monthly)
    loss_et_monthly = hh.compute_nan_mse_time_2(pred = et_predicted_monthly, target = et_target_monthly)
    loss_q_monthly = hh.compute_nan_mse_time_2(pred = q_predicted_monthly, target = q_target_monthly)
    loss_swe = hh.compute_nan_mse_time_2(pred = swe_predicted_monthly, target = swe_target_monthly)
    loss_fapar_monthly = hh.compute_nan_mse_time(pred = fapar_predicted_monthly, target = fapar_target_monthly)
    loss_gpp_monthly = hh.compute_nan_mse_time_2(pred = gpp_predicted_monthly, target = gpp_target_monthly)
    loss_nee_monthly = hh.compute_nan_mse_time(pred = nee_predicted_monthly, target = nee_target_monthly)
    loss_nee_iav_monthly = self.compute_mse(nee_iav_predicted_avg, nee_iav_target_avg)
    loss_cue_ma = hh.compute_nan_mse_static(cue_ma_predicted_standardized, cue_ma_values_standardized, loss_fn = self.compute_mse)

    # add penalty for the learned constants
    loss_prior_deviation_beta_co2 = hh.prior_deviation_loss(estimate = learned_constants["beta_co2"], 
                                                            prior = torch.tensor(15 / (100 * 100)),
                                                            prior_std = torch.tensor(5 / (100 * 100)))
    loss_prior_deviation_q10 = hh.prior_deviation_loss(estimate = learned_constants["beta_q10"], 
                                                       prior = torch.tensor(1.5),
                                                       prior_std = torch.tensor(0.3))

    # add all the losses together
    loss_sum = loss_tws_monthly + loss_et_monthly + loss_q_monthly + loss_swe + \
        loss_fapar_monthly + loss_gpp_monthly + loss_nee_monthly + loss_nee_iav_monthly + loss_cue_ma + \
        loss_prior_deviation_beta_co2 + loss_prior_deviation_q10

    # store all the individual losses (inc. sum) together in a dictionary
    losses_all = {"tws": loss_tws_monthly,
                  "et": loss_et_monthly,
                  "runoff": loss_q_monthly,
                  "swe": loss_swe,
                  "fapar": loss_fapar_monthly,
                  "gpp": loss_gpp_monthly,
                  "nee": loss_nee_monthly,
                  "nee_iav": loss_nee_iav_monthly,
                  "cue": loss_cue_ma,
                  "beta_co2": loss_prior_deviation_beta_co2,
                  "beta_q10": loss_prior_deviation_q10,
                  "loss_sum": loss_sum
                  }

    # return the final loss
    return losses_all