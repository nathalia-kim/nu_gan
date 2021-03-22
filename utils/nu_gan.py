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
#experiment_root = '/Users/kim/experiment/'
experiment_root = 'C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/experiment/'
positive_images_root = experiment_root + 'data/original/positive_images/' 
negative_images_root = experiment_root + 'data/original/negative_images/' 
positive_npy_root = experiment_root + 'data/segmented/positive_npy/'
negative_npy_root = experiment_root + 'data/segmented/negative_npy/'
ref_path = experiment_root + 'data/original/reference/BM_GRAZ_HE_0007_01.png'

#cell_level_data
X_train_path = experiment_root + 'data/cell_level_label/X_train.npy' 
X_test_path = experiment_root + 'data/cell_level_label/X_test.npy' 
y_train_path = experiment_root + 'data/cell_level_label/y_train.npy' 
y_test_path = experiment_root + 'data/cell_level_label/y_test.npy' 

#n_epoch = 50 # number of epochs
n_epoch = 3
batchsize = 10
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
    cell_representation(X_train_path, X_test_path, y_train_path, y_test_path, 
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