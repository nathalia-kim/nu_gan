import os
import time
import argparse
from experiment import cell_segmentation, cell_representation, image_classification

parser = argparse.ArgumentParser()
parser.add_argument('--task', 
                    choices = ['cell_representation', 'image_classification', 'cell_segmentation'], 
                    help='cell_representation | image_classification | cell_segmentation')
opt = parser.parse_args()
opt.task = 'cell_representation'

if not (opt.task):
    parser.error("specify a task such as '--task cell_representation'")

# for image classification and nuclei segmentation
#experiment_root = 'C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/'
experiment_root = "/Users/kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/"
positive_images_root = experiment_root + 'experiment/data/original/positive_images/' 
negative_images_root = experiment_root + 'experiment/data/original/negative_images/' 
positive_npy_root = experiment_root + 'experiment/data/segmented/positive_npy/'
negative_npy_root = experiment_root + 'experiment/data/segmented/negative_npy/'
ref_path = experiment_root + 'experiment/data/original/reference/BM_GRAZ_HE_0007_01.png'

# cell_level_data
dataset = 'A'

if dataset == 'A':
    X_train_path = experiment_root + 'dataset_A/cell_level_label/X_train.npy' 
    X_test_path = experiment_root + 'dataset_A/cell_level_label/X_test.npy' 
    y_train_path = experiment_root + 'dataset_A/cell_level_label/y_train.npy' 
    y_test_path = experiment_root + 'dataset_A/cell_level_label/y_test.npy' 
    
if dataset == 'B':
    X_train_path = experiment_root + 'dataset_B/cell_level_label/X_train.npy' 
    X_test_path = experiment_root + 'dataset_B/cell_level_label/X_test.npy' 
    y_train_path = experiment_root + 'dataset_B/cell_level_label/y_train.npy' 
    y_test_path = experiment_root + 'dataset_B/cell_level_label/y_test.npy'

n_epoch = 10 #50 # number of epochs
batchsize = 10 #10
rand = 32 # number of gaussian noise variables 
dis_category = 5 # number of categories / clusters
ld = 1e-4 # learning rate for discriminator network D
lg = 1e-4 # learning rate for generator network G
lq = 1e-4 # learning rate for auxiliary network Q
random_seed = 42
save_model_steps = 100 # number of steps to save the model 
intensity = 160 # intensity for segmentation thresholding 
multi_process = True # multi core process for nuclei segmentation

fold = 4
choosing_fold = 1 # cross-validation for classification

time = str(int(time.time()))
if 1- os.path.exists(experiment_root + time):
    os.makedirs(experiment_root + time)
    os.makedirs(experiment_root + time + '/' + 'picture')
    os.makedirs(experiment_root + time + '/' + 'model')
    
experiment_root = experiment_root + time + '/'
print('folder_name:' + str(time))

if opt.task == 'cell_representation':
    values_D_G, l_q, purities = cell_representation(X_train_path, X_test_path, 
                                                    y_train_path, y_test_path, 
                        experiment_root, n_epoch, batchsize, rand, dis_category, 
                        ld, lg, lq, save_model_steps)

if opt.task == 'image_classification':
    image_classification(positive_images_root, negative_images_root, 
                         positive_npy_root,negative_npy_root, ref_path, intensity, 
                         X_train_path, X_test_path, y_train_path, y_test_path, 
                         experiment_root, multi_process, fold, random_seed, 
                         choosing_fold, n_epoch, batchsize, rand, dis_category, 
                         ld, lg, lq, save_model_steps)

if opt.task == 'cell_segmentation':
    cell_segmentation(positive_images_root, negative_images_root, positive_npy_root, 
                          negative_npy_root, ref_path, intensity, multi_process)

#%% evaluate results 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

# sample purities to plot
sample_purities = []
indexes = []
for i in range(0, len(purities)):
    if i % 25 == 0:
        sample_purities.append(purities[i])
        indexes.append(i)     
figure(figsize=(20, 12), dpi=80)
plt.plot(indexes, sample_purities)

# convert l_q to numpy
l_q_np = []
for i in range(0, len(l_q)):
    l_q_np.append(l_q[i].cpu().detach().numpy())
    
# sample l_q to plot
sample_lq = []
indexes = []
for i in range(0, len(l_q_np)):
    if i % 25 == 0:
        sample_lq.append(l_q_np[i])
        indexes.append(i)     
figure(figsize=(20, 12), dpi=80)
plt.plot(indexes, sample_lq)

output_dir = experiment_root + '/'
np.save(output_dir + 'purities', purities)
np.save(output_dir + 'values_D_G', values_D_G)
np.save(output_dir + 'l_q', l_q_np)

#%% visualize cluster assignments - Recreate figure 8

from generate_figures import figure_8
experiment_id = 1617143319
experiment_root = 'C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/'

# pth file names
netD_fn = "netD_0.787468671679198_0.922397332160461_14600.pth"
netG_fn = "netG_0.787468671679198_0.922397332160461_14600.pth"
netD_Q_fn = "netD_Q_0.787468671679198_0.922397332160461_14600.pth"
netD_D_fn = "netD_D_0.787468671679198_0.922397332160461_14600.pth"

figure_8(X_train_path, X_test_path, experiment_id, experiment_root, netD_fn, netG_fn, netD_Q_fn, netD_D_fn, rand = 32, dis_category = 5)



