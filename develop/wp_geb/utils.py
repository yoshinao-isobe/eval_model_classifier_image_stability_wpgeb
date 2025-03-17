import gzip
import numpy as np
import math


def load_dataset_list(
        labels_file_name, images_file_name,
        image_width, image_height, color_size,
        dataset_size):

    print('Load labels: ', labels_file_name)
    labels = load_gz(labels_file_name)
    print('Load images: ', images_file_name)
    images = load_gz(images_file_name)

    labels = labels.ravel()
    images = images.ravel()

    images = images.reshape(-1, image_width, image_height, color_size)

    loaded_dataset_size = labels.shape[0]

    if loaded_dataset_size < dataset_size:
        dataset_size = loaded_dataset_size

    images = images[:dataset_size]
    labels = labels[:dataset_size]

    return images, labels


def split_nary(nary1, nary2, block_size):
    size = nary2.shape[0]

    if block_size > size or block_size == 0:
        block_size = size

    dataset_list = [
        (nary1[i: i + block_size], nary2[i: i + block_size])
        for i in range(0, size, block_size)]

    return dataset_list


# load dataset or model
def load_gz(fn):
    with gzip.open(fn, "rb") as f:
        loaded_data = np.load(f)
    return loaded_data


# save text
def save_message(fn, message, mode):  # mode = 'w' or 'a'
    f = open(fn, mode)
    f.write(message)
    f.close()


# ------------------------------------------------------------
#   set trainable attribute to non_trainable in layer_name
# ------------------------------------------------------------

def set_non_trainable_layer(model, layer_name):
    layers = model.layers
    for i in range(len(layers)):
        layer = layers[i]
        weights = layer.trainable_weights
        for j in range(len(weights)):
            if layer.__class__.__name__ == layer_name:
                layer.trainable = False
    return model


def model_trainable_params_size(model):
    return params_size(model.trainable_weights)


def params_size(params):
    # tr_size = np.sum([np.prod(v.get_shape()) for v in params])
    tr_size = np.sum([np.prod(v.shape) for v in params])
    total_size = int(tr_size)
    return total_size


# ---------------------------------
# numerical computation
# ---------------------------------

# kl^(-1)(q,c)
def inv_binary_kl_div(q, c, eps_nm, max_nm):
    if q == 0:
        p = 1.0 - math.exp(-c)
        return to_float(p)

    else:
        p = q + math.sqrt(c / 2.0)

        for i in range(max_nm):
            if p >= 1.0:
                return 1.0

            if math.fabs(p - c) < eps_nm:
                return to_float(p)

            h1 = binary_kl_div(q, p) - c
            h2 = (1 - q) / (1 - p) - q / p
            p = p - h1 / h2

        return to_float(p)


def binary_kl_div(q, p):
    kl = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))

    return kl


def to_float(p):
    if p.__class__.__name__ == 'EagerTensor':
        return p.numpy()
    else:
        return p

