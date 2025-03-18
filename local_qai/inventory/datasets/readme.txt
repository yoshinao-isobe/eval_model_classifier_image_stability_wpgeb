# make small datasets of MNIST and CIFAR10 for demanstration of this AIT

 (e.g., python==3.10.16)

 pip install -r requirements.txt
 python make_test_datasets_gz.py

# It generates the following datasets:

- mnist/images_test_1000.gz, labels_test_1000.gz

  This dataset is a part (1000 images) of the MNIST database,
  originally created by Yann LeCun, Corinna Cortes, and Christopher
  Burges. The dataset is licensed under the Creative Commons
  Attribution-Share Alike 3.0 License (CC BY-SA 3.0). The original
  dataset can be found at http://yann.lecun.com/exdb/mnist/.

- cifar10/images_test_1000.gz, labels_test_1000.gz

  This dataset is a part (1000 images) of the CIFAR-10 test-dataset,
  originally created by Alex Krizhevsky, Vinod Nair, and Geoffrey
  Hinton. The dataset is licensed under the MIT License. The original
  dataset is available at https://www.cs.toronto.edu/~kriz/cifar.html.


