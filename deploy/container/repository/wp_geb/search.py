# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)
# searching for adversarial perturbations

import numpy as np
import tensorflow as tf
import wp_geb.utils as utl


# ===================================================
#    FGSM (Fast Gradient-Sign Method on Weights)
# ===================================================
def search(model, images, labels, search_block_size, perturb_ratio):

    dataset_list = utl.split_nary(images, labels, search_block_size)

    offset = 0
    err_id_list = []
    for (in_dataset, out_dataset) in dataset_list:

        sub_err_ids = eval_err_fgsm_sub(
            model, in_dataset, out_dataset, perturb_ratio)

        err_id_list += [sub_err_id + offset for sub_err_id in sub_err_ids]
        offset += len(out_dataset)

    return err_id_list


def eval_err_fgsm_sub(
        model, in_dataset, out_dataset, perturb_ratio):

    gradients = pre_example_gradients(model, in_dataset, out_dataset)
    grad_params = model.trainable_variables
    for i in range(len(gradients)):
        gradients[i] = - perturb_ratio * np.abs(grad_params[i]) * np.sign(gradients[i])

    err_id_list = eval_sum_err_grad_list(model, in_dataset, out_dataset, gradients, grad_params)
    return err_id_list


def eval_sum_err_grad_list(model, in_dataset, out_dataset, gradients, grad_params):
    layers = model.layers
    org_weight = [lyr.get_weights() for lyr in layers]

    # err_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    err_opt = tf.keras.optimizers.SGD(learning_rate=1.0)

    data_size = len(in_dataset)
    grad_size = len(gradients)
    gradients_list = [list(gradients[i]) for i in range(grad_size)]

    err_id_list = []
    for i in range(data_size-1, -1, -1):

        grads = [gradients_list[j].pop() for j in range(grad_size)]
        grads_and_vars = zip(grads, grad_params)
        err_opt.apply_gradients(grads_and_vars)

        err = eval_single_error(model, in_dataset[i], out_dataset[i])
        if err == 1:
            err_id_list += [i]

        for j in range(len(layers)):
            layers[j].set_weights(org_weight[j])

    err_id_list.sort()

    return err_id_list


# --------------------
#  Single error
# --------------------
def eval_single_error(model, idata, odata):
    tf_idata = tf.expand_dims(idata, 0)
    predict = model(tf_idata)
    predicted_class = tf.cast(tf.argmax(predict, axis=1), tf.uint8)
    err1 = tf.cast(tf.math.not_equal(odata, predicted_class), tf.int32)
    err = err1[0]
    return err


# --------------------
#    gradients
# --------------------

@tf.function(reduce_retracing=True)
def single_gradients(model, in_data, out_data):
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy()

    in_data0 = tf.expand_dims(in_data, 0)
    out_data0 = tf.expand_dims(out_data, 0)
    params = model.trainable_variables

    with tf.GradientTape() as tape:
        # tape.watch(params)
        out_predict1 = model(in_data0)
        neg_loss = loss_fun(out_data0, out_predict1)

    grad = tape.gradient(neg_loss, params)

    return grad


# if perturb_bn is 0 then the gradients in layer "name_BN" is 0.
@tf.function(reduce_retracing=True)
def pre_example_gradients(model, in_dataset, out_dataset):

    def pre_single_gradients(in_out_data):
        in_data, out_data = in_out_data
        grad = single_gradients(model, in_data, out_data)
        return grad

    gradients = tf.vectorized_map(pre_single_gradients, (in_dataset, out_dataset))
    return gradients


# --------------------
#  Single margin
# --------------------

def eval_single_margin(model, idata, odata):

    tf_idata = tf.expand_dims(idata, 0)
    predict = model(tf_idata)
    predict0 = predict[0]

    if type(odata) is np.ndarray:
        odata = odata[0]
    correct_prob = predict0[odata]
    probs, indices = tf.math.top_k(predict0, k=2)
    if correct_prob == probs[0]:
        other_prob = probs[1]
    else:
        other_prob = probs[0]

    margin = correct_prob - other_prob

    return margin.numpy()


