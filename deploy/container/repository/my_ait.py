#!/usr/bin/env python
# coding: utf-8

# # AIT Development notebook

# ## notebook of structure

# | #  | Name                                               | cells | for_dev | edit               | description                                                                |
# |----|----------------------------------------------------|-------|---------|--------------------|----------------------------------------------------------------------------|
# | 1  | [Environment detection](##1-Environment-detection) | 1     | No      | uneditable         | detect whether the notebook are invoked for packaging or in production     |
# | 2  | [Preparing AIT SDK](##2-Preparing-AIT-SDK)         | 1     | Yes     | uneditable         | download and install AIT SDK                                               |
# | 3  | [Dependency Management](##3-Dependency-Management) | 3     | Yes     | required(cell #2)  | generate requirements.txt for Docker container                             |
# | 4  | [Importing Libraries](##4-Importing-Libraries)     | 2     | Yes     | required(cell #1)  | import required libraries                                                  |
# | 5  | [Manifest Generation](##5-Manifest-Generation)     | 1     | Yes     | required           | generate AIT Manifest                                                      |
# | 6  | [Prepare for the Input](##6-Prepare-for-the-Input) | 1     | Yes     | required           | generate AIT Input JSON (inventory mapper)                                 |
# | 7  | [Initialization](##7-Initialization)               | 1     | No      | uneditable         | initialization for AIT execution                                           |
# | 8  | [Function definitions](##8-Function-definitions)   | N     | No      | required           | define functions invoked from Main area.<br> also define output functions. |
# | 9  | [Main Algorithms](##9-Main-Algorithms)             | 1     | No      | required           | area for main algorithms of an AIT                                         |
# | 10 | [Entry point](##10-Entry-point)                    | 1     | No      | uneditable         | an entry point where Qunomon invoke this AIT from here                     |
# | 11 | [License](##11-License)                            | 1     | Yes     | required           | generate license information                                               |
# | 12 | [Deployment](##12-Deployment)                      | 1     | Yes     | uneditable         | convert this notebook to the python file for packaging purpose             |

# ## notebook template revision history

# 1.0.1 2020/10/21
# 
# * add revision history
# * separate `create requirements and pip install` editable and noeditable
# * separate `import` editable and noeditable
# 
# 1.0.0 2020/10/12
# 
# * new cerarion

# ## body

# ### #1 Environment detection

# [uneditable]

# In[1]:


# Determine whether to start AIT or jupyter by startup argument
import sys
is_ait_launch = (len(sys.argv) == 2)


# ### #2 Preparing AIT SDK

# [uneditable]

# In[2]:


if not is_ait_launch:
    # get ait-sdk file name
    from pathlib import Path
    from glob import glob
    import re
    import os

    current_dir = get_ipython().run_line_magic('pwd', '')

    ait_sdk_path = "./ait_sdk-*-py3-none-any.whl"
    ait_sdk_list = glob(ait_sdk_path)
    ait_sdk_name = os.path.basename(ait_sdk_list[-1])

    # install ait-sdk
    get_ipython().system('pip install -q --upgrade pip')
    get_ipython().system('pip install -q --no-deps --force-reinstall ./$ait_sdk_name')


# ### #3 Dependency Management

# #### #3-1 [uneditable]

# In[3]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_requirements_generator import AITRequirementsGenerator
    requirements_generator = AITRequirementsGenerator()


# #### #3-2 [required]

# In[4]:


if not is_ait_launch:
     requirements_generator.add_package('tensorflow', '2.16.2')
     requirements_generator.add_package('numpy', '1.26.4')


# #### #3-3 [uneditable]

# In[5]:


if not is_ait_launch:
    requirements_generator.add_package(f'./{ait_sdk_name}')
    requirements_path = requirements_generator.create_requirements(current_dir)

    get_ipython().system('pip install -q -r $requirements_path ')


# ### #4 Importing Libraries

# #### #4-1 [required]

# In[6]:


import os
import math
import time
import numpy as np
import tensorflow as tf
import wp_geb.utils as utl
import wp_geb.search as sch
import wp_geb.measure as msr
import wp_geb.estimate as est


# #### #4-2 [uneditable]

# In[7]:


# must use modules
from os import path
import shutil  # do not remove
from ait_sdk.common.files.ait_input import AITInput  # do not remove
from ait_sdk.common.files.ait_output import AITOutput  # do not remove
from ait_sdk.common.files.ait_manifest import AITManifest  # do not remove
from ait_sdk.develop.ait_path_helper import AITPathHelper  # do not remove
from ait_sdk.utils.logging import get_logger, log, get_log_path  # do not remove
from ait_sdk.develop.annotation import measures, resources, downloads, ait_main  # do not remove
# must use modules


# ### #5 Manifest Generation

# [required]

# In[8]:


if not is_ait_launch:
     from ait_sdk.common.files.ait_manifest_generator import AITManifestGenerator
     manifest_genenerator = AITManifestGenerator(current_dir)
     manifest_genenerator.set_ait_name('eval_model_classifier_image_stability_wpgeb')
     manifest_genenerator.set_ait_description('This AIT is a simplified version of WP-GEB-Estimator-2, designed to estimates weight-perturbed generalization bounds. It can be used for evaluating the statistically certified stability of neural classifiers. The manual can be downloaded from the web-site of WP-GEB-Estimator (https://staff.aist.go.jp/y-isobe/wp-geb-estimator/).')
     manifest_genenerator.set_ait_source_repository('https://github.com/aistairc/eval_model_classifier_image_stability_wpgeb')
     manifest_genenerator.set_ait_version('0.1')
     manifest_genenerator.add_ait_licenses('Apache License Version 2.0')
     manifest_genenerator.add_ait_keywords('generalization risk')
     manifest_genenerator.set_ait_quality('https://ait-hub.pj.aist.go.jp/ait-hub/api/0.0.1/qualityDimensions/機械学習品質マネジメントガイドライン第三版/C-2機械学習モデルの安定性')
     inventory_requirement_data = manifest_genenerator.format_ait_inventory_requirement(format_=['gz'])
     inventory_requirement_model = manifest_genenerator.format_ait_inventory_requirement(format_=['h5'])
     manifest_genenerator.add_ait_inventories(name='images_gz', 
                                              type_='dataset', 
                                              description='The list (gzipped flat numpy array) of the images in the test dataset. Note: the image data are directly input to the classifier without any normalization. For example, if a classifier is trained by the normalized gray scale (0, 1) instead of (0, 255) in MNIST, then the image file with (already normalized) gray scale (0, 1) has to be loaded in this AIT.',
                                              requirement=inventory_requirement_data)
     manifest_genenerator.add_ait_inventories(name='labels_gz', 
                                              type_='dataset',
                                              description='The list (gzipped flat numpy array) of the labels in the test dataset',
                                              requirement=inventory_requirement_data)
     manifest_genenerator.add_ait_inventories(name='model_h5', 
                                              type_='model',
                                              description='The evaluated model (classifier) by the test dataset',
                                              requirement=inventory_requirement_model)
     manifest_genenerator.add_ait_parameters(name='dataset_size',
                                             type_='int', 
                                             description='dataset size used for testing', 
                                             default_val='1000')
     manifest_genenerator.add_ait_parameters(name='image_width',
                                             type_='int', 
                                             description='width of each image', 
                                             default_val='32')
     manifest_genenerator.add_ait_parameters(name='image_height',
                                             type_='int',
                                             description='height of each image', 
                                             default_val='32')
     manifest_genenerator.add_ait_parameters(name='color_size',
                                             type_='int',
                                             description='color size of each image', 
                                             default_val='3')
     manifest_genenerator.add_ait_parameters(name='search_batch_size',
                                             type_='int',
                                             description='batch size for parallel search', 
                                             default_val='20')
     manifest_genenerator.add_ait_parameters(name='prediction_batch_size',
                                             type_='int',
                                             description='batch size for parallel prediction', 
                                             default_val='200')
     manifest_genenerator.add_ait_parameters(name='perturb_ratio',
                                             type_='float',
                                             description='ratio of maximum weight-perturbation to weight', 
                                             default_val='0.001')
     manifest_genenerator.add_ait_parameters(name='skip_search',
                                             type_='int',
                                             description='1 if gradient-based search is skipped (0 otherwise)',
                                             default_val='0')
     manifest_genenerator.add_ait_parameters(name='err_threshold',
                                             type_='float',
                                             description='acceptable error threshold',
                                             default_val='0.01')
     manifest_genenerator.add_ait_parameters(name='confidence',
                                             type_='float',
                                             description='confidence of generalization bounds',
                                             default_val='0.9')
     manifest_genenerator.add_ait_measures(name='gen_err_ub', 
                                           type_='float', 
                                           description='generalization error upper bound (no weight-perturbation)',
                                           structure='single',
                                           min='0',
                                           max='1')
     manifest_genenerator.add_ait_measures(name='test_err', 
                                           type_='float', 
                                           description='test error (no weight-perturbation)', 
                                           structure='single',
                                           min='0',
                                           max='1')
     manifest_genenerator.add_ait_measures(name='wp_gen_risk_ub', 
                                           type_='float', 
                                           description='weight-perturbed generalization risk upper bound', 
                                           structure='single',
                                           min='0',
                                           max='1')
     manifest_genenerator.add_ait_measures(name='wp_test_risk_ub', 
                                           type_='float', 
                                           description='weight-perturbed test risk upper bound', 
                                           structure='single',
                                           min='0',
                                           max='1')
     manifest_genenerator.add_ait_measures(name='wp_gen_err_ub', 
                                           type_='float', 
                                           description='weight-perturbed generalization error upper bound', 
                                           structure='single',
                                           min='0',
                                           max='1')
     manifest_genenerator.add_ait_measures(name='wp_test_err_ub', 
                                           type_='float', 
                                           description='weight-perturbed test error upper bound', 
                                           structure='single',
                                           min='0',
                                           max='1')
     manifest_genenerator.add_ait_resources(name='estimation_result',  
                                           type_='text', 
                                           description='estimation result')
     manifest_genenerator.add_ait_resources(name='input_output_data',
                                           type_='table', 
                                           description='input data and output data')
     manifest_genenerator.add_ait_downloads(name='Log', 
                                            description='AIT log')
     manifest_path = manifest_genenerator.write()


# ### #6 Prepare for the Input

# [required]

# In[9]:


if not is_ait_launch:
     from ait_sdk.common.files.ait_input_generator import AITInputGenerator
     input_generator = AITInputGenerator(manifest_path)
     input_generator.add_ait_inventories(name='images_gz',
                                         value='datasets/cifar10/images_test_1000.gz')
     input_generator.add_ait_inventories(name='labels_gz',
                                         value='datasets/cifar10/labels_test_1000.gz')
     input_generator.add_ait_inventories(name='model_h5',
                                         value='models/cifar10/cnn_m.h5')
     input_generator.write()


# ### #7 Initialization

# [uneditable]

# In[10]:


logger = get_logger()

ait_manifest = AITManifest()
ait_input = AITInput(ait_manifest)
ait_output = AITOutput(ait_manifest)

if is_ait_launch:
    # launch from AIT
    current_dir = path.dirname(path.abspath(__file__))
    path_helper = AITPathHelper(argv=sys.argv, ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)
else:
    # launch from jupyter notebook
    # ait.input.json make in input_dir
    input_dir = '/usr/local/qai/mnt/ip/job_args/1/1'
    current_dir = get_ipython().run_line_magic('pwd', '')
    path_helper = AITPathHelper(argv=['', input_dir], ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)

ait_input.read_json(path_helper.get_input_file_path())
ait_manifest.read_json(path_helper.get_manifest_file_path())

### do not edit cell


# ### #8 Function definitions

# [required]

# In[11]:


@log(logger)
@measures(ait_output, 'gen_err_ub')
def out_gen_err_ub(gen_err_ub):
    return gen_err_ub

@log(logger)
@measures(ait_output, 'test_err')
def out_test_err(test_err):
    return test_err

@log(logger)
@measures(ait_output, 'wp_gen_risk_ub')
def out_wp_gen_risk_ub(wp_gen_risk_ub):
    return wp_gen_risk_ub

@log(logger)
@measures(ait_output, 'wp_test_risk_ub')
def out_wp_test_risk_ub(wp_test_risk_ub):
    return wp_test_risk_ub

@log(logger)
@measures(ait_output, 'wp_gen_err_ub')
def out_wp_gen_err_ub(wp_gen_err_ub):
    return wp_gen_err_ub

@log(logger)
@measures(ait_output, 'wp_test_err_ub')
def out_wp_test_err_ub(wp_test_err_ub):
    return wp_test_err_ub


# In[12]:


@log(logger)
@resources(ait_output, path_helper, 'estimation_result', 'estimation_result.txt')
def save_estimation_result(txt, file_path: str=None) -> str:
    with open(file_path, 'w') as f:
        f.write(txt)

@log(logger)
@resources(ait_output, path_helper, 'input_output_data', 'input_output_data.csv')
def save_input_output_data(csv, file_path: str=None) -> str:
    with open(file_path, 'w') as f:
        f.write(csv)


# In[13]:


## sample ##
@log(logger)
@downloads(ait_output, path_helper, 'Log', 'ait.log')
def move_log(file_path: str=None) -> str:
    shutil.move(get_log_path(), file_path)


# ### #9 Main Algorithms

# [required]

# In[14]:


@log(logger)
@ait_main(ait_output, path_helper, is_ait_launch)
def main() -> None:

    # constant
    perturb_batch_norm = 0
    batch_normalization_name = 'BatchNormalization'
    not_available = 'N/A'
    delta0_ratio = 0.5

    # parameters
    param_dataset_size = ait_input.get_method_param_value('dataset_size')
    param_image_width = ait_input.get_method_param_value('image_width')
    param_image_height = ait_input.get_method_param_value('image_height')
    param_color_size = ait_input.get_method_param_value('color_size')
    param_search_batch_size = ait_input.get_method_param_value('search_batch_size')
    param_prediction_batch_size = ait_input.get_method_param_value('prediction_batch_size')
    param_perturb_ratio = ait_input.get_method_param_value('perturb_ratio')
    param_skip_search = ait_input.get_method_param_value('skip_search')
    param_err_threshold = ait_input.get_method_param_value('err_threshold')
    param_confidence = ait_input.get_method_param_value('confidence')
    
    delta = 1 - param_confidence

    # input
    images_file_name = ait_input.get_inventory_path('images_gz')
    labels_file_name = ait_input.get_inventory_path('labels_gz')
    model_file_name = ait_input.get_inventory_path('model_h5')

    # --- load dataset ---
    images, labels = utl.load_dataset_list(
        labels_file_name, images_file_name,
        param_image_width, param_image_height, param_color_size,
        param_dataset_size)
    dataset_size = labels.shape[0]

    # --- load model ---
    print('Load model: ', model_file_name)
    model = tf.keras.models.load_model(model_file_name)
    # model.summary()
    if perturb_batch_norm == 0:
        model = utl.set_non_trainable_layer(model, batch_normalization_name)
    p_params_size = utl.model_trainable_params_size(model)

    time1 = time.time()

    dataset_list = utl.split_nary(images, labels, param_prediction_batch_size)
    test_err, tmp_id_list = msr.error_evaluation_batch(model, dataset_list)

    # --- gradient-based search for risky data ---
    if param_skip_search == 1 or param_perturb_ratio == 0:
        err_id_list = []
        err_num_search = 0
        print('Gradient-based search: skipped')
    else:
        print('Gradient-based search')
        err_id_list = sch.search(
            model, images, labels, param_search_batch_size, param_perturb_ratio)
        err_num_search = len(err_id_list)
        # info_str = 'Detected risky data: {:d}/{:d}'.format(
        #     err_num_search, dataset_size)
        # print(info_str)
        # print('err_id_list = ', err_id_list)

    # --- measure errs with random weight perturbations ---
    print('Random perturbation')
            
    if len(err_id_list) > 0:
        images = np.delete(images, np.array(err_id_list), axis=0)
        labels = np.delete(labels, np.array(err_id_list))

    nd_dataset_size = labels.shape[0]

    if nd_dataset_size == 0 or param_perturb_ratio == 0:
        err_num_random = 0
        practical_err_thr = 0
        perturb_sample_size = 0
        wp_test_err = test_err

    else:
        delta0 = delta * delta0_ratio
        delta1 = delta0 / nd_dataset_size
        perturb_sample_size = math.ceil(math.log(delta1, 1 - param_err_threshold))
        practical_err_thr = 1 - math.exp(-math.log(1 / delta1) / perturb_sample_size)

        err_count = msr.measure(
            model, images, labels, param_prediction_batch_size,
            param_perturb_ratio, perturb_sample_size)

        wp_test_err = np.sum(err_count) / (nd_dataset_size * perturb_sample_size)

        err_flag = (err_count > 0).astype(int)
        err_num_random = np.sum(err_flag)

    # --- generalization ---
    print('Generalization')
    
    delta0 = delta * delta0_ratio

    (gen_err_ub,
     wp_gen_risk_ub, wp_test_risk_ub, conf_risk, conf0_risk,
     non_det_rate_ub, gen_err_thr_ub,
     wp_gen_err_ub, wp_test_err_ub, conf_err, conf0_err,
     avl_err) = est.estimate(
        dataset_size, err_num_search, err_num_random, param_perturb_ratio,
        test_err, delta, delta0, param_err_threshold, perturb_sample_size)

    time2 = time.time()
    elapsed_time = time2 - time1

    # make estimation result message
    info_str = ''
    info_str += 'Model file: ' + model_file_name + '\n'
    info_str += 'Image file: ' + images_file_name + '\n'
    info_str += 'Label file: ' + labels_file_name + '\n\n'

    info_str += 'Normal Test (without weight-perturbation):\n'
    info_str += '  Generalization error bound: '
    info_str += '{:.2f}% (Conf: {:g}%)\n'.format(
        gen_err_ub * 100, (1 - delta) * 100)
    info_str += '  Test error: '
    info_str += '{:.2f}%\n'.format(
        test_err * 100)

    if param_perturb_ratio > 0:
        info_str += 'Test with weight-perturbation (perturbation ratio: {:g}):\n'.format(
            param_perturb_ratio)

        if param_skip_search == 0:
            info_str += '  Gradient-based search\n'
            info_str += '    Detected risky data: {:d}/{:d}\n'.format(
                err_num_search, dataset_size)
            info_str += '    Generalization non-detection rate bound: {:.2f}% (Conf: {:g}%)\n'.format(
                non_det_rate_ub * 100, (1 - delta) * 100)

        info_str += '  Random perturbation (sample size: {:d})\n'.format(
            perturb_sample_size)

        # gen-ub (risk adapt)
        if err_num_search == 0:
            info_str += '    Estimated risk (without search):\n'
        else:
            info_str += '    Estimated risk (with search):\n'
        info_str += '      Perturbed generalization risk bound: '
        info_str += '{:.2f}% (Conf: {:g}%)\n'.format(
            wp_gen_risk_ub * 100, conf_risk * 100)
        info_str += '      Perturbed test risk bound: '
        info_str += '{:.2f}% (Conf: {:g}%)\n'.format(
            wp_test_risk_ub * 100, conf0_risk * 100)
        info_str += '        Generalization acceptable error threshold bound: '
        info_str += '{:g}% (Conf: {:g}%)\n'.format(
            gen_err_thr_ub * 100, conf_risk * 100)
        info_str += '        Individual acceptable error threshold: '
        info_str += '{:g}% (Practical: {:.4f}%)\n'.format(
            param_err_threshold * 100, practical_err_thr * 100)

        if avl_err:
            # gen-ub (err)
            info_str += '    Estimated error:\n'
            info_str += '      Perturbed generalization error bound: '
            info_str += '{:.2f}% (Conf: {:g}%)\n'.format(
                wp_gen_err_ub * 100, conf_err * 100)
            info_str += '      Perturbed test error bound: '
            info_str += '{:.2f}% (Conf: {:g}%)\n'.format(
                wp_test_err_ub * 100, conf0_err * 100)
            info_str += '        Sample-perturbed test error: '
            info_str += '{:.2f}% \n'.format(wp_test_err * 100)

    info_str += '\nThe meaning of the results is as follows:\n\n'
    info_str += ('Test error, which is misclassification rate, of the neural classifier '
                 'given by Model file ')
    info_str += 'is {:.2f}% '.format(test_err * 100)
    info_str += 'for the dataset given by Image file and Label file. '
    info_str += ('Here, it is assumed that the dataset is sampled '
                 'according to an independent and identical distribution (i.i.d). ')
    info_str += ('The generalization error, '
                 'which is the expected value of error '
                 'for any input-image (i.e., including unseen images) sampled '
                 'according to the distribution, '
                 'is less than {:.2f}% at least {:g}% confidence.\n\n').format(
        gen_err_ub * 100, (1 - delta) * 100)

    info_str += 'Hereafter, the ratio of weight-perturbation to weight is less than {:g}. '.format(
            param_perturb_ratio)
    if avl_err:
        info_str += ('The weight-perturbed generalization error for any image and '
                     'any weight-perturbation '
                     'is less than {:.2f}% at least {:g}% confidence. ').format(
            wp_gen_err_ub * 100, conf_err * 100)
    info_str += ('The weight-perturbed generalization risk, '
                 'which is the expected value of the probability such that '
                 'randomly sampled input-image is risky, '
                 'is less than {:.2f}% at least {:g}% confidence. ').format(
        wp_gen_risk_ub * 100, conf_risk * 100)
    info_str += ('Here, an input-image is said to be risky if the misclassification-rate '
                 'caused by weight-perturbations for the image exceeds '
                 'the acceptable error threshold which is less than {:g}%. ').format(
        param_err_threshold * 100)
    info_str += ('Generalization acceptance error threshold is the expected value of'
                 'the acceptable error threshold for any input-image and it is less than'
                 '{:.2f}% at least {:g}% confidence. \n\n').format(
        gen_err_thr_ub * 100, conf_risk * 100)

    info_str += '(Elapsed time for estimation: {:.1f} [sec])\n'.format(elapsed_time)

    # make input output data (csv)
    info_csv = ''
    info_csv += 'images_file_name, labels_file_name, model_file_name, '
    info_csv += 'dataset_size, image_width, image_height, color_size, '
    info_csv += 'search_batch_size, prediction_batch_size, perturb_ratio, '
    info_csv += 'skip_search, err_threshold, confidence, '

    info_csv += 'perturbed_params, '
    info_csv += 'gen_err_ub, test_err, err_num_search, '
    info_csv += 'non_det_rate_ub, perturb_sample_size, practical_err_threshold, '

    info_csv += 'wp_gen_risk_ub, wp_test_risk_ub, conf_risk, conf0_risk, '
    info_csv += 'gen_err_thr_ub, '
    info_csv += 'wp_gen_err_ub, wp_test_err_ub, conf_err, conf0_err\n'

    info_csv += str(images_file_name) + ', ' + str(labels_file_name) + ', ' + str(model_file_name) + ', '
    info_csv += str(param_dataset_size) + ', ' + str(param_image_width) + ', '
    info_csv += str(param_image_height) + ', ' + str(param_color_size) + ', '
    info_csv += str(param_search_batch_size) + ', ' + str(param_prediction_batch_size)  + ', '
    info_csv += str(param_perturb_ratio) + ', '
    info_csv += str(param_skip_search) + ', ' + str(param_err_threshold) + ', '
    info_csv += str(param_confidence) + ', '

    info_csv += str(p_params_size) + ', '
    info_csv += str(gen_err_ub) + ', ' + str(test_err) + ', '

    if param_skip_search == 0:
        info_csv += str(err_num_search) + ', ' + str(non_det_rate_ub) + ', '
    else:
        info_csv += not_available + ', ' + not_available + ', '

    info_csv += str(perturb_sample_size) + ', ' + str(practical_err_thr) + ', '

    info_csv += str(wp_gen_risk_ub) + ', '
    info_csv += str(wp_test_risk_ub) + ', ' + str(conf_risk) + ', ' + str(conf0_risk) + ', '
    info_csv += str(gen_err_thr_ub) + ', '

    if avl_err:
        info_csv += str(wp_gen_err_ub) + ', ' + str(wp_test_err_ub) + ', '
        info_csv += str(conf_err) + ', ' + str(conf0_err) + '\n'
    else:
        info_csv += not_available + ', ' + not_available + ', '
        info_csv += not_available + ', ' + not_available + '\n'

    # save
    save_estimation_result(info_str)
    save_input_output_data(info_csv)
    
    # measures
    out_gen_err_ub(gen_err_ub)
    out_test_err(test_err)
    out_wp_gen_risk_ub(wp_gen_risk_ub)
    out_wp_test_risk_ub(wp_test_risk_ub)
    out_wp_gen_err_ub(wp_gen_err_ub)
    out_wp_test_err_ub(wp_test_err_ub)
    
    move_log()


# ### #10 Entry point

# [uneditable]

# In[ ]:


if __name__ == '__main__':
    main()


# ### #11 License

# [required]

# In[16]:


ait_owner='AIST'
ait_creation_year='2025'


# ### #12 Deployment

# [uneditable] 

# In[17]:


if not is_ait_launch:
    from ait_sdk.deploy import prepare_deploy
    from ait_sdk.license.license_generator import LicenseGenerator
    
    current_dir = get_ipython().run_line_magic('pwd', '')
    prepare_deploy(ait_sdk_name, current_dir, requirements_path)
    
    # output License.txt
    license_generator = LicenseGenerator()
    license_generator.write('../top_dir/LICENSE.txt', ait_creation_year, ait_owner)


# In[ ]:





# In[ ]:




