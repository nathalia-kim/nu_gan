from gan_model import get_matrix, compute_purity_entropy, get_f_score, create_model, create_loader, normalized
from generate_figures import figure_8
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
        "--dis_category",
        default=5,
        type=int,
        help="number of categories / clusters"
    )

parser.add_argument(
        "--model_path",
        default="C:/Users/Kim/OneDrive - Queen's University/Courses/CISC-867 Deep Learning/Project/1617921325/model/",
        type=str,
        help="path with pretrained models"
    )

opt = parser.parse_args()

experiment_root = "./"

model_path = opt.model_path
X_train_path = experiment_root + 'dataset_A/cell_level_label/X_train.npy' 
X_test_path = experiment_root + 'dataset_A/cell_level_label/X_test.npy' 
y_train_path = experiment_root + 'dataset_A/cell_level_label/y_train.npy' 
y_test_path = experiment_root + 'dataset_A/cell_level_label/y_test.npy'

netD_fn="netD_best_model.pth"
netG_fn="netG_best_model.pth"
netD_Q_fn="netD_Q_best_model.pth"
netD_D_fn="netD_D_best_model.pth"
rand=32
dis_category=opt.dis_category

# instantiate models 
netD, netG, netD_D, netD_Q = create_model(rand=rand, dis_category=dis_category)

print("Loading models...")

# load model from pth file 
netD.load_state_dict(torch.load(model_path + netD_fn))
netD_Q.load_state_dict(torch.load(model_path + netD_Q_fn))

print("Done")

 # get test data
# load training and testing datasets
X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = np.load(y_train_path)
y_test = np.load(y_test_path)

cell_train_set = np.concatenate([X_train, X_test])
cell_test_set = cell_train_set
cell_test_label = np.concatenate([y_train, y_test])

test = normalized(cell_test_set)
test_loader = create_loader(test, shuffle=False, batchsize=1)
test_label = cell_test_label

print("Evaluating...")

# compute confusion matrix, entropy, purity and f-score
confusion_matrix = get_matrix(netD, netD_Q, test_loader,
                          test_label, dis_category)
entropy, purity = compute_purity_entropy(confusion_matrix)
f_score = get_f_score(confusion_matrix)
print("Done")
print('purity:', purity, 'entropy:', entropy, 'f_score', f_score)

figure_8(X_train_path, X_test_path, model_path, output_dir=model_path)
print("Saved visualization of results!")

   