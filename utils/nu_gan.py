import os
import time
import argparse
from experiment import cell_segmentation, cell_representation, image_classification, cell_representation_unlabeled
from cell_classification import cell_classification
from generate_figures import figure_8
import warnings
warnings.filterwarnings('always')

parser = argparse.ArgumentParser()
parser.add_argument('--task', 
                    choices = ['cell_representation', 'image_classification', 'cell_segmentation', 'cell_representation_unlabeled', 'cell_classification'], 
                    help='cell_representation | image_classification | cell_segmentation | cell_representation_unlabeled | cell_classification')
opt = parser.parse_args()
#opt.task = 'cell_representation'

if not (opt.task):
    parser.error("specify a task such as '--task cell_representation'")

# for image classification and nuclei segmentation
#experiment_root = 'C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/'
experiment_root = "./"
positive_images_root = experiment_root + 'experiment/data/original/positive_images/' 
negative_images_root = experiment_root + 'experiment/data/original/negative_images/' 
positive_npy_root = experiment_root + 'experiment/data/segmented/positive_npy/'
negative_npy_root = experiment_root + 'experiment/data/segmented/negative_npy/'
ref_path = experiment_root + 'experiment/data/original/reference/BM_GRAZ_HE_0007_01.png'

# dataset A
X_train_path = experiment_root + 'dataset_A/cell_level_label/X_train.npy' 
X_test_path = experiment_root + 'dataset_A/cell_level_label/X_test.npy' 
y_train_path = experiment_root + 'dataset_A/cell_level_label/y_train.npy' 
y_test_path = experiment_root + 'dataset_A/cell_level_label/y_test.npy' 

# unlabeled dataset
images_path = "C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/new dataset/TCGA_data/blca/"
ref_path = "C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/new dataset/TCGA_data/reference/"
npy_path = "C:/Users/Kim/OneDrive - Queen\'s University/Courses/CISC-867 Deep Learning/Project/new dataset/TCGA_data/"

n_epoch = 100 # number of epochs
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

# for cell classification
experiment_id = 1617143319

time = str(int(time.time()))
if 1- os.path.exists(experiment_root + time):
    os.makedirs(experiment_root + time)
    os.makedirs(experiment_root + time + '/' + 'picture')
    os.makedirs(experiment_root + time + '/' + 'model')
    
experiment_root = experiment_root + time + '/'
print('folder_name:' + str(time))

if opt.task == 'cell_segmentation':
    cell_segmentation(positive_images_root, negative_images_root, positive_npy_root, 
                          negative_npy_root, ref_path, intensity, multi_process)

if opt.task == 'cell_representation':
    values_D_G, l_q, purities = cell_representation(X_train_path, X_test_path, 
                                                    y_train_path, y_test_path, 
                        experiment_root, n_epoch, batchsize, rand, dis_category, 
                        ld, lg, lq, save_model_steps)
    
if opt.task == 'cell_representation_unlabeled':
    values_D_G, l_q = cell_representation_unlabeled(
        images_path, ref_path, npy_path, experiment_root, n_epoch, batchsize, rand, 
        dis_category, ld, lg, lq, save_model_steps)
    
    # view resulting representations
    figure_8(X_train_path, X_test_path, experiment_root)
    
if opt.task == 'cell_classification':
    max_iter = 3000
    project_root = os.path.abspath(os.path.join(experiment_root, os.pardir))
    cell_classification(experiment_id, project_root, X_train_path, X_test_path, y_train_path, y_test_path, max_iter, rand, dis_category)

if opt.task == 'image_classification':
    values_D_G, l_q, purities = image_classification(positive_images_root, negative_images_root, 
                         positive_npy_root,negative_npy_root, ref_path, intensity, 
                         X_train_path, X_test_path, y_train_path, y_test_path, 
                         experiment_root, multi_process, fold, random_seed, 
                         choosing_fold, n_epoch, batchsize, rand, dis_category, 
                         ld, lg, lq, save_model_steps)
