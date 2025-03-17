# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# estimating generalization risk/error upper bounds with perturbations on weights

import math
import wp_geb.utils as utl
# import utils as utl


def estimate(dataset_size, err_num_search, err_num_random, perturb_ratio,
             test_err, delta, delta0, err_thr, perturb_sample_size):
    newton_eps_err = 1e-6
    newton_itr_max = 10
    # not_available = 'N/A'

    # --- test error upper bound without perturbation ---
    # by Chernoff bound
    kl_ub = math.log(1.0 / delta) / dataset_size
    gen_err_np_ub = utl.inv_binary_kl_div(
        test_err, kl_ub, newton_eps_err, newton_itr_max)

    err_num = err_num_random + err_num_search

    if dataset_size == err_num:
        # test_risk = 1.0
        test_risk_ub = 1.0
        gen_risk_ub = 1.0
        gen_err_thr_ub = 0
        conf_risk = 1.0
        conf0_risk = 1.0
        non_det_rate_ub = 0

    elif perturb_ratio == 0:
        delta_ge = delta
        conf0_risk = 1
        # by Chernoff bound
        kl_ub = math.log(1.0 / delta_ge) / dataset_size
        gen_risk_ub = utl.inv_binary_kl_div(
            test_err, kl_ub, newton_eps_err, newton_itr_max)
        conf_risk = 1.0 - delta
        test_risk_ub = test_err

        gen_err_thr_ub = 0
        non_det_rate_ub = 1

    else:  # err_num < datasize

        # upper bound of generalized acceptable threshold
        non_det_rate = 1 - err_num_search / dataset_size
        kl_ub = math.log(1.0 / delta) / dataset_size
        non_det_rate_ub = utl.inv_binary_kl_div(
            non_det_rate, kl_ub, newton_eps_err, newton_itr_max)
        gen_err_thr_ub = err_thr * non_det_rate_ub

        # upper bound of test risk rate
        test_risk_ub = err_num / dataset_size
        delta_ge = delta - delta0
        kl_ub = math.log(1.0 / delta_ge) / dataset_size
        gen_risk_ub = utl.inv_binary_kl_div(
            test_risk_ub, kl_ub, newton_eps_err, newton_itr_max)

        # confidence
        conf_risk = 1.0 - delta
        conf0_risk = 1.0 - delta0

    # ----------------------------------------
    # generalization error bound
    # ----------------------------------------

    # search has been skipped
    if err_num_search == 0:
        avl_err = True

        if perturb_ratio == 0:
            delta_ge = delta
            conf0_err = 1
            # by Chernoff bound
            # kl_ub = math.log(1.0 / delta_ge) / datasize
            # gen_err_ub = utl.inv_binary_kl_div(test_err, kl_ub, params.eps_nm, params.max_nm)
            gen_err_ub = gen_risk_ub
            conf_err = 1.0 - delta
            test_err_ub = test_err

        else:
            # by Perez-Ortiz 2021
            kl_ub_err = math.log(1.0 / delta0) / perturb_sample_size
            test_err_ub = utl.inv_binary_kl_div(
                test_err, kl_ub_err, newton_eps_err, newton_itr_max)

            # upper bound of the generalization error
            delta_ge = delta - delta0
            # by Maurer bound 2004
            kl_ub = math.log(2 * math.sqrt(dataset_size) / delta_ge) / dataset_size
            gen_err_ub = utl.inv_binary_kl_div(
                test_err_ub, kl_ub, newton_eps_err, newton_itr_max)

            conf0_err = 1 - delta0
            conf_err = 1 - delta

    # random perturbation is not available.
    else:
        avl_err = False

        # test_err = 0
        test_err_ub = 1
        gen_err_ub = 1
        conf0_err = 0
        conf_err = 0

    # -----------------------------
    # save and print results
    # -----------------------------

    return (
        gen_err_np_ub,
        gen_risk_ub, test_risk_ub, conf_risk, conf0_risk,
        non_det_rate_ub, gen_err_thr_ub,
        gen_err_ub, test_err_ub, conf_err, conf0_err,
        avl_err
    )
