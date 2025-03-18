import gzip
import time
import threading
import numpy as np
import tensorflow as tf
import os

Reload_max = 10
Reload_wait_time = 5
lock = threading.Lock()

root = '.'
dataset_names = ['mnist', 'cifar10']
sub_name = 'test'
dataset_size = 1000


def main():

    for dataset_name in dataset_names:
        dr = root + '/' + dataset_name
        print('dr = ', dr)

        if not os.path.isdir(dr):
            os.mkdir(dr)
            print('make: ', dr)

        if dataset_name == 'mnist':
            dataset_module = tf.keras.datasets.mnist
            image_size = 28 * 28
            # 28 x 28 x 1
        else:  # dataset_name == 'cifar10':
            dataset_module = tf.keras.datasets.cifar10
            image_size = 32 * 32 * 3
            # 32 x 32 x 3

        (train_images, train_labels), (images, labels) = reload_data(dataset_module)

        print(f'---{dataset_name}---')

        images = images / 255.0
        images = images.ravel()

        labels = labels.ravel()
        loaded_dataset_size = labels.shape[0]
        if loaded_dataset_size > dataset_size:
            images = images[:dataset_size * image_size]
            labels = labels[:dataset_size]

        # .gz format
        images_fn = dr + '/images_' + sub_name + '_' + str(dataset_size) + '.gz'
        labels_fn = dr + '/labels_' + sub_name + '_' + str(dataset_size) + '.gz'

        # save images by numpy+gzip format
        with gzip.open(images_fn, "wb") as f:
            np.save(f, images)

        # save labels by numpy+gzip format
        with gzip.open(labels_fn, "wb") as f:
            np.save(f, labels)


def reload_data(dataset_module):
    emp = np.array([])
    for n in range(Reload_max):
        try:
            with lock:
                (train_in_dataset, train_out_dataset), \
                    (test_in_dataset, test_out_dataset) = \
                    dataset_module.load_data()
            return (train_in_dataset, train_out_dataset), (test_in_dataset, test_out_dataset)
        except Exception as e:
            print(f"Loading dataset was failed --> retry [{n+1}]: {e}")
            time.sleep(Reload_wait_time)

    print("Loading dataset was failed. --> abandon")
    return (emp, emp), (emp, emp)


if __name__ == '__main__':
    main()

