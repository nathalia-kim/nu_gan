import os
from sklearn.model_selection import KFold
import numpy as np
from segmentation_functions import cell_segment, masks_to_npy
from gan_model import create_model, rotation, train_representation
import multiprocessing
from generate_figures import figure_8
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def cell_segmentation(positive_images_root, negative_images_root, positive_npy_root, 
                      negative_npy_root, ref_path, intensity, multi_core):
    '''
    Performs cell segmentation on input images

    Parameters
    ----------
    positive_images_root : str
        path with positive images.
    negative_images_root : str
        path with negative images.
    positive_npy_root : str
        path with positive npy file.
    negative_npy_root : str
        path with negative npy file.
    ref_path : str
        path with reference image for stain normalization.
    intensity : int
        intensity for segmentation thresholding.
    multi_core : bool, optional
        if the process is multi core. The default is True.

    Returns
    -------
    None.

    '''
    
    # get paths of positive and negative images
    positive_images_path = [positive_images_root + n for n in 
                            os.listdir(positive_images_root)]
    negative_images_path = [negative_images_root + n for n in 
                            os.listdir(negative_images_root)]
    
    # create directories
    if 1- os.path.exists(positive_npy_root + str(intensity) + '/'):
        os.makedirs(positive_npy_root + str(intensity) + '/')
    if 1- os.path.exists(negative_npy_root + str(intensity) + '/'):
        os.makedirs(negative_npy_root + str(intensity) + '/')
    
    # apply cell segmentation on images 
    if (multi_core == True and __name__ == '__main__'):
        jobs = []
        
        for index, i in enumerate(positive_images_path):
            p = multiprocessing.Process(
                target=cell_segment, args=(i, positive_npy_root + str(intensity) +
                                           '/', ref_path, intensity))
            p.start()
            jobs.append(p)
            if (index + 1) % 7 == 0:
                p.join()
                jobs = []

        for job in jobs:
            p.join()

        jobs = []
        for index, i in enumerate(negative_images_path):
            p = multiprocessing.Process(
                target=cell_segment, args=(i, negative_npy_root + str(intensity) 
                                           + '/', ref_path, intensity))
            p.start()
            jobs.append(p)
            if (index + 1) % 7 == 0:
                p.join()
                jobs = []
            
        for job in jobs:
            p.join()
            
    else:
        for index, i in enumerate(positive_images_path):
            cell_segment(i, positive_npy_root + str(intensity) 
                         + '/', ref_path, intensity)
        for index, i in enumerate(negative_images_path):
            cell_segment(i, negative_npy_root + str(intensity)
                         + '/', ref_path, intensity)

def split_dataset(path, fold=4, random_seed=42):
    '''
    Split dataset in k folds

    Parameters
    ----------
    path : str
        path to npy file with images.
    fold : int, optional
        number of folds to split the dataset. The default is 4.
    random_seed : int, optional
        random seed. The default is 42.

    Returns
    -------
    train_list : list
        list with paths of split training data files.
    test_list : list
        list with paths of split testing data files.

    '''
    np.random.seed(random_seed)
    kf = KFold(n_splits=fold, shuffle=True)
    kf.get_n_splits(path)
    train_list, test_list  = [], []
    
    for train_index, test_index in kf.split(path):
        train_list.append([path[n] for n in train_index])
        test_list.append([path[n] for n in test_index])
        
    return train_list, test_list

def eval_plots(values_D_G, l_q, purities, experiment_root, sample=30):
    '''
    Create and save evaluation plots - purity and loss over iterations

    Parameters
    ----------
    values_D_G : array
       values of function V(D, G) over iterations. Used to evaluate how well the generator distribution matches the real data distribution.
    l_q : array
        values of loss function of auxiliary network over iterations.
    purities : array
        values of clustering purity over iterations.
    experiment_root : str
        path of experiment root.
    sample : int, optional
        sample rate to plot. The default is 30.

    Returns
    -------
    None.

    '''
    if len(purities) > 0:
        # sample purities to plot
        sample_purities = []
        indexes = []
        for i in range(0, len(purities)):
            if i % 30 == 0:
                sample_purities.append(purities[i])
                indexes.append(i)     
        figure(figsize=(10, 6), dpi=80)
        plt.plot(indexes, sample_purities, label="purities")
        plt.xlabel("Generator iterations")
        plt.ylabel("Purity")
        plt.legend()
        plt.savefig(experiment_root + "purities.png")
        
    # sample l_q and vdg to plot
    sample_lq = []
    sample_vdg = []
    indexes = []
    for i in range(0, len(l_q)):
        if i % 30 == 0:
            sample_lq.append(l_q[i])
            sample_vdg.append(values_D_G[i])
            indexes.append(i)     
    figure(figsize=(10, 6), dpi=80)
    plt.plot(indexes, sample_vdg, label="V(D,G)")
    plt.plot(indexes, sample_lq, label="Lq")
    plt.xlabel("Generator iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(experiment_root + "loss.png")
    

def cell_representation(X_train_path, X_test_path, y_train_path, y_test_path, 
                        experiment_root, n_epoch=50, batchsize=16, rand=32, 
                        dis_category=5, ld = 1e-4, lg = 1e-4, lq = 1e-4, 
                        save_model_steps=100, image_classification = False):
    '''
    Creates and trains model of cell-level visual representation learning
    
    Parameters
    ----------
    X_train_path : str
        path to .npy file with training data 
    X_test_path : str
        path to .npy file with testing data 
    y_train_path : str
        path to .npy file with training labels
    y_test_path : str
        path to .npy file with testing labels 
    experiment_root : str
        path to experiment root
    n_epoch : int
        number of epochs for training. The default is 50.
    batchsize : int
        batch size. The default is 16.
    rand : int
        number of gaussian noise variables. The default is 32.
    dis_category : int
        number of categories / clusters. The default is 5.
    ld : float
        learning rate for discriminator network D. The default is 1e-4.
    lg : float
        learning rate for generator network G. The default is 1e-4.
    lq : float
        learning rate for auxiliary network Q. The default is 1e-4.
    save_model_steps : int
        number of steps to save the model. The default is 100.
    image_classification : bool, optional
        if the training is for image classification or not. The default is False.

    Returns
    -------
    values_D_G : list
        values of function V(D, G) over iterations. Used to evaluate how well the generator distribution matches the real data distribution 
    l_q : list
        values of loss function of auxiliary network over iterations.
    purities: list
        values of clustering purity over iterations

    '''
    
    # load training and testing datasets
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    # create cell training and testing sets 
    cell_train_set = np.concatenate([X_train, X_test])
    cell_test_set = cell_train_set
    cell_test_label = np.concatenate([y_train, y_test])
    
    # initialize empty npys
    positive_train_npy = []
    positive_test_npy = [] 
    negative_train_npy = [] 
    negative_test_npy = []
    
    # create / initialize the model 
    netD, netG, netD_D, netD_Q = create_model(rand=rand, dis_category=dis_category)
    
    # train cell representation
    values_D_G, l_q, purities = train_representation(
                         cell_train_set, cell_test_set, cell_test_label, 
                         positive_train_npy, positive_test_npy, negative_train_npy, 
                         negative_test_npy, netD, netG,                      
                         netD_D, netD_Q, experiment_root, n_epoch=n_epoch, 
                         batchsize=batchsize, rand=rand, 
                         dis_category=dis_category, ld=ld, lg=lg, lq=lq, 
                         save_model_steps=save_model_steps, 
                         image_classification = image_classification)
    
    # convert l_q to numpy
    l_q_np = []
    for i in range(0, len(l_q)):
        l_q_np.append(l_q[i].cpu().detach().numpy())
    
    # save results
    np.save(experiment_root + 'purities', purities)
    np.save(experiment_root + 'values_D_G', values_D_G)
    np.save(experiment_root + 'l_q', l_q_np)
    
    # save evaluation plots
    eval_plots(values_D_G, l_q_np, purities, experiment_root)
    
    # view resulting representations
    model_path = experiment_root + "/model/"
    figure_8(X_train_path, X_test_path, model_path, output_dir=experiment_root)
    
    return values_D_G, l_q_np, purities
    

def cell_representation_unlabeled(images_path, ref_path, npy_path, experiment_root, n_epoch=50, 
                                  batchsize=16, rand=32, dis_category=5, ld = 1e-4, lg = 1e-4, 
                                  lq = 1e-4, save_model_steps=100, image_classification = False):
    '''
    Creates and trains model of cell-level visual representation learning with unlabeled data 

    Parameters
    ----------
    images_path : str
        path with images for preping.
    ref_path : str
        path with reference image for stain normalization.
    npy_path : str
        path to save npy file with single-cell images.
    experiment_root : str
        path to experiment root
    n_epoch : int
        number of epochs for training. The default is 50.
    batchsize : int
        batch size. The default is 16.
    rand : int
        number of gaussian noise variables. The default is 32.
    dis_category : int
        number of categories / clusters. The default is 5.
    ld : float
        learning rate for discriminator network D. The default is 1e-4.
    lg : float
        learning rate for generator network G. The default is 1e-4.
    lq : float
        learning rate for auxiliary network Q. The default is 1e-4.
    save_model_steps : int
        number of steps to save the model. The default is 100.
    image_classification : bool, optional
        if the training is for image classification or not. The default is False.

    Returns
    -------
    values_D_G : list
        values of function V(D, G) over iterations. Used to evaluate how well the generator distribution matches the real data distribution 
    l_q : list
        values of loss function of auxiliary network over iterations.

    '''
    
    # prep data, generate npy file
    masks_to_npy(images_path, ref_path, npy_path)
    
    # load training data
    X_train = np.load(npy_path + "Train.npy")
    
    # create datasets
    cell_train_set = X_train
    cell_test_set = np.array([])
    cell_test_label = np.array([])
    
    # initialize empty npys
    positive_train_npy = []
    positive_test_npy = [] 
    negative_train_npy = [] 
    negative_test_npy = []
    
    # create / initialize the model 
    netD, netG, netD_D, netD_Q = create_model(rand=rand, dis_category=dis_category)
    
    # train cell representation
    values_D_G, l_q = train_representation(
                         cell_train_set, cell_test_set, cell_test_label, 
                         positive_train_npy, positive_test_npy, negative_train_npy, 
                         negative_test_npy, netD, netG,                      
                         netD_D, netD_Q, experiment_root, n_epoch=n_epoch, 
                         batchsize=batchsize, rand=rand, 
                         dis_category=dis_category, ld=ld, lg=lg, lq=lq, 
                         save_model_steps=save_model_steps, 
                         image_classification = image_classification)
    
    # convert l_q to numpy
    l_q_np = []
    for i in range(0, len(l_q)):
        l_q_np.append(l_q[i].cpu().detach().numpy())
    
    # save results
    np.save(experiment_root + 'values_D_G', values_D_G)
    np.save(experiment_root + 'l_q', l_q_np)
    
    # save evaluation plots
    eval_plots(values_D_G, l_q_np, [], experiment_root)
    
    X_train_path = npy_path + "Train.npy"
    X_test_path = None
    model_path = experiment_root + "/model/"
    figure_8(X_train_path, X_test_path, model_path, rand = rand, dis_category = dis_category, output_dir=experiment_root)
    
    return values_D_G, l_q_np

def image_classification(positive_images_root, negative_images_root, 
                         positive_npy_root,negative_npy_root, ref_path, intensity, 
                         X_train_path, X_test_path, y_train_path, y_test_path, 
                         experiment_root, multi_core = True, fold = 4, random_seed=42, 
                         choosing_fold = 1, n_epoch=10000, batchsize=32, rand=64, 
                         dis_category=5, ld = 1e-4, lg = 1e-4, lq = 1e-4, 
                         save_model_steps = 100, image_classification = True):
    '''
    Applies cell segmentation to images. Creates and trains model of cell-level visual representation learning. Performs image classification 

    Parameters
    ----------
    positive_images_root : str
        path with positive images.
    negative_images_root : str
        path with negative images.
    positive_npy_root : str
        path with positive npy file.
    negative_npy_root : str
        path with negative npy file.
    ref_path : str
        path with reference image for stain normalization.
    intensity : int
        intensity for segmentation thresholding.
    X_train_path : str
        path with training data.
    X_test_path : str
        path with testing data.
    y_train_path : str
        path with training labels.
    y_test_path : str
        path with testing labels.
    experiment_root : str
        path of experiment root.
    multi_core : bool, optional
        if the process is multi core. The default is True.
    fold : int, optional
        number of folds to split dataset. The default is 4.
    random_seed : int, optional
        random seed. The default is 42.
    choosing_fold : int, optional
        The default is 1.
    n_epoch : int, optional
        number of epochs for training. The default is 10000.
    batchsize : int, optional
        size of the batch. The default is 32.
    rand : int, optional
        number of gaussian noise variables. The default is 64.
    dis_category : int, optional
        number of categories / clusters. The default is 5.
    ld : float, optional
        learning rate for discriminator network D. The default is 1e-4.
    lg : float, optional
        learning rate for generator network G. The default is 1e-4.
    lq : float, optional
        learning rate for auxiliary network Q. The default is 1e-4.
    save_model_steps : int, optional
        number of steps to save the model. The default is 100.
    image_classification : bool, optional
        if the training is for image classification or not. The default is True.

    Returns
    -------
    None.

    '''

    positive_npy_path = [positive_npy_root +str(intensity)+'/' + n[:-3] + 
                         'npy' for n in os.listdir(positive_images_root)]
    negative_npy_path =[negative_npy_root +str(intensity)+'/' + n[:-3] + 
                        'npy' for n in os.listdir(negative_images_root)]

    positive_train_list, positive_test_list = split_dataset(positive_npy_path, 
                                                            fold, random_seed)
    negative_train_list, negative_test_list = split_dataset(negative_npy_path, 
                                                            fold, random_seed)

    positive_train_npy = [np.load(n, allow_pickle=True) for n in 
                          positive_train_list[choosing_fold]]
    positive_test_npy = [np.load(n, allow_pickle=True) for n in 
                         positive_test_list[choosing_fold]]
    negative_train_npy = [np.load(n, allow_pickle=True) for n in 
                          negative_train_list[choosing_fold]]
    negative_test_npy = [np.load(n, allow_pickle=True) for n in 
                         negative_test_list[choosing_fold]]

    cell_train_set = np.concatenate([np.concatenate(positive_train_npy), np.concatenate(negative_train_npy)])
    
    # load training and testing datasets
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    # create cell training and testing sets
    cell_test_set = np.concatenate([X_train, X_test])
    cell_test_label = np.concatenate([y_train, y_test])
    cell_train_set = rotation(cell_test_set)
    
    # create / initialize the model 
    netD, netG, netD_D, netD_Q = create_model(rand=rand, dis_category=dis_category)
    
    # train cell representation
    values_D_G, l_q, purities = train_representation(
        cell_train_set, cell_test_set, cell_test_label, positive_train_npy, 
        positive_test_npy, negative_train_npy, negative_test_npy, netD, netG, netD_D, 
        netD_Q, experiment_root, n_epoch=n_epoch, batchsize=batchsize, rand=rand, 
        dis_category=dis_category,
        ld = ld, lg = lg, lq = lq, save_model_steps=save_model_steps, 
        image_classification=image_classification,)
    
    # convert l_q to numpy
    l_q_np = []
    for i in range(0, len(l_q)):
        l_q_np.append(l_q[i].cpu().detach().numpy())
    
    # save results
    np.save(experiment_root + 'purities', purities)
    np.save(experiment_root + 'values_D_G', values_D_G)
    np.save(experiment_root + 'l_q', l_q_np)
    
    # save evaluation plots
    eval_plots(values_D_G, l_q_np, purities, experiment_root)
    
    # view resulting representations
    model_path = experiment_root + "/model/"
    figure_8(X_train_path, X_test_path, model_path, output_dir=experiment_root)
    
    return values_D_G, l_q_np, purities