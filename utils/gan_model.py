import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable
from imgaug import augmenters as iaa
from sklearn.cluster import KMeans
import math
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

class avgpool(nn.Module):
    '''
    Mean pooling class - downsampling
    '''
    def __init__(self, up_size=0):
        super(avgpool, self).__init__()
        
    def forward(self, x):
        out_man = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4
        return out_man

class ResidualBlock(nn.Module):
    '''
    Residual block class
    3 types: upsample, downsample, None 
    '''

    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock, self).__init__()
        if resample == 'up':
            self.bn1 = nn.BatchNorm2d(in_dim)
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.upsample = torch.nn.Upsample(up_size,2)
            self.upsample = torch.nn.Upsample(scale_factor=2)
            self.upsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(out_dim)
            
        elif resample == 'down':
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.pool = avgpool()
            self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
        
        elif resample == None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            
        self.resample = resample

    def forward(self, x):
        
        if self.resample == None:
            shortcut = x
            output = x
            
            output = nn.functional.relu(output)
            output = self.conv1(output)
            output = nn.functional.relu(output)
            output = self.conv2(output)
            
        elif self.resample == 'up':
            shortcut = x
            output = x
            
            shortcut = self.upsample(shortcut) #upsampleconv
            shortcut = self.upsample_conv(shortcut)
            
            output = self.bn1(output)
            output = nn.functional.relu(output)
            output = self.conv1(output)

            output = self.bn2(output)
            output = nn.functional.relu(output)
            output = self.upsample(output) #upsampleconv
            output = self.conv2(output)
                        
        elif self.resample == 'down':
            shortcut = x
            output = x
            
            shortcut = self.pool_conv(shortcut) #convmeanpool
            shortcut = self.pool(shortcut)
            
            output = nn.functional.relu(output)
            output = self.conv1(output)
            
            output = nn.functional.relu(output)
            output = self.conv2(output)    #convmeanpool
            output = self.pool(output)
            
        return output+shortcut

class ResidualBlock_thefirstone(nn.Module):
    '''
    First residual block class 
    '''
    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock_thefirstone, self).__init__()
        
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
        self.pool = avgpool()
        self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
        
    def forward(self, x):
        
        shortcut = x
        output = x
        
        shortcut = self.pool(shortcut) #meanpoolconv
        shortcut = self.pool_conv(shortcut)

        output = self.conv1(output)
        output = nn.functional.relu(output)
        output = self.conv2(output) #convmeanpool
        output = self.pool(output)
            
        return output+shortcut

class generator(nn.Module):
    '''
    Generator network G class 
    3 upsampling residual blocks
    '''

    def __init__(self, rand=128):
        super(generator, self).__init__()
        self.rand = rand
        self.linear = nn.Linear(rand, 2048, bias=True)
        self.layer_up_1 = ResidualBlock(128, 128, 'up', up_size=8)  
        self.layer_up_2 = ResidualBlock(128, 128, 'up', up_size=16)
        self.layer_up_3 = ResidualBlock(128, 128, 'up', up_size=32)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_last = nn.Conv2d(128, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x = x.view(-1, self.rand)
        x = self.linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.layer_up_1(x)
        x = self.layer_up_2(x)
        x = self.layer_up_3(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv_last(x)
        self.tanh = nn.Tanh()
        return x

class discriminator(nn.Module):
    '''
    First part of discriminator network D class
    4 residual blocks, 1 downsampling 
    '''

    def __init__(self):
        super(discriminator, self).__init__()
        self.layer_down_1 = ResidualBlock_thefirstone(3, 128)
        self.layer_down_2 = ResidualBlock(128, 128, 'down')
        self.layer_none_1 = ResidualBlock(128, 128, None)
        self.layer_none_2 = ResidualBlock(128, 128, None)
        
    def forward(self, x):
        x = self.layer_down_1(x)
        x = self.layer_down_2(x)
        x = self.layer_none_1(x)
        x = self.layer_none_2(x)
        x = nn.functional.relu(x)
        x = x.mean(2).mean(2)
        return x
    
class _netD_D(nn.Module):
    '''
    Second part of discriminator network D
    '''
    def __init__(self):
        super(_netD_D, self).__init__()
        self.linear = nn.Linear(128,1, bias=True)
        
    def forward(self, x):
        x = x.view(-1, 128)
        x = self.linear(x)
        return x.view(-1,1,1,1)
    
class _netD_Q(nn.Module):
    '''
    Second part of auxiliary network Q
    '''
    def __init__(self, nd = 10):
        super(_netD_Q, self).__init__()
        self.linear = nn.Linear(128,nd, bias=True)
        self.softmax = nn.LogSoftmax()
        self.nd = nd

    def forward(self, x):
        x = x.view(-1, 128)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, -1)
        return x.view(-1,self.nd,1,1)

def uniform(stdev, size):
    '''

    Parameters
    ----------
    stdev : float
        standard deviation
    size : int
        output shape

    Returns
    -------
    ndarray
        Drawn samples from the parameterized uniform distribution

    '''
    return np.random.uniform(
                low = -stdev * np.sqrt(3),
                high = stdev * np.sqrt(3),
                size = size
            ).astype('float32')

def initialize_conv(m):
    '''

    Parameters
    ----------
    m : torch neural network module 

    Returns
    -------
    filter_values : ndarray
        Weight values initialization for convolutional layers 

    '''
    fan_in = m.in_channels * m.kernel_size[0]**2
    fan_out = m.out_channels * m.kernel_size[0]**2 / (m.stride[0]**2)

    if m.kernel_size[0] == 3:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    # Normalized init (Glorot & Bengio)
    else: 
        filters_stdev = np.sqrt(2./(fan_in+fan_out))
        
    filter_values = uniform(filters_stdev,
                            (m.kernel_size[0], m.kernel_size[0], m.in_channels, 
                             m.out_channels))
    
    return filter_values

def initialize_linear(m):
    '''

    Parameters
    ----------
    m : torch neural network module

    Returns
    -------
    weight_values : ndarray
        Weight values initialization for linear layers 

    '''
    weight_values = uniform(np.sqrt(2./(m.in_features + m.out_features)),
                            (m.in_features, m.out_features))
    return weight_values

def weights_init(m):
    '''
    Applies weight initialization recursively to every submodule of module m
    
    Parameters
    ----------
    m : torch neural network module

    Returns
    -------
    None.

    '''
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        weight = torch.from_numpy(initialize_conv(m))
        m.weight.data.copy_ = weight
        m.bias.data.fill_(0)
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
    elif classname.find('Linear') != -1:
        weight_values = torch.from_numpy(initialize_linear(m))
        m.weight.data.copy_ = weight_values
        m.bias.data.fill_(0)
        
def sample_c(batchsize=32, dis_category=5):
    '''
    Draw samples from a categorical / multinomial distribution

    Parameters
    ----------
    batchsize : int
        Size of the batch. The default is 32.
    dis_category : int, optional
        number of categories / clusters. The default is 5.

    Returns
    -------
    rand_c : tensor
        Tensor with random samples. Shape: [batchsize, discategory]
    label_c : tensor
        Labels of random samples. Shape: [batchsize]

    '''
    # initialize rand_c with zeros 
    rand_c = np.zeros((batchsize, dis_category), dtype='float32')
    
    for i in range(0, batchsize):
        rand = np.random.multinomial(1, dis_category*[1/float(dis_category)], size=1)
        rand_c[i] = rand
    
    # get labels by finding max value per sample
    label_c = np.argmax(rand_c, axis=1)
    # construct tensors
    label_c = torch.LongTensor(label_c.astype('int'))
    rand_c = torch.from_numpy(rand_c.astype('float32'))
    return rand_c, label_c
        
def fix_noise(rand=128, dis_category=5, row=10):
    '''
    Get fixed noise from standard normal distribution

    Parameters
    ----------
    rand : int, optional
        number of noise variables. The default is 128.
    dis_category : int, optional
        number of categories / clusters. The default is 5.
    row : int, optional
        number of rows. The default is 10.

    Returns
    -------
    Array of float
        Fixed noise, shape: (dis_category*row, dis_category + rand, 1, 1).

    '''
    
    # get samples of the standard normal distribution. Shape: [dis_category*row, rand]
    fixed_z = np.random.randn(row, rand).repeat(dis_category, axis=0)
    
    # initialize changing_dis with zeros. Shape: [row*dis_category, dis_category]
    changing_dis = np.zeros((row*dis_category, dis_category), dtype = np.float32)
    
    # Shape: [row*dis_category]
    list = [n for n in range(0, dis_category)]*row
    
    for i in range(0, row*dis_category):
        changing_dis[i, list[i]] = 1
    
    # shape: [dis_category*row, dis_category + rand]
    map1 = np.concatenate((changing_dis, fixed_z), axis=1)
    lst = [map1.astype(np.float32)]
    
    # shape: (dis_category*row, dis_category + rand, 1, 1)
    return lst[0].reshape(row*dis_category, rand+dis_category, 1, 1)

def calc_gradient_penalty(netD_D, netD, real_data, fake_data, lamda, batch_size):
    '''
    Compute gradient penalty to add to the discriminator, enforce Lipschitz 
    constraint

    Parameters
    ----------
    netD_D : _netD_D object
        module type of netD_D network.
    netD : discriminator object
        module type of discriminator network.
    real_data : tensor
        real input data. Shape: (batchsize, 3, image_height, image_width)
    fake_data : tensor
        fake input data. Shape: (batchsize, 3, image_height, image_width)
    lamda : int
        hyperparameter in the loss of the discriminator.
    batch_size : int
        size of the batch.

    Returns
    -------
    gradient_penalty : tensor 
        tensor containing gradient penalty value (float).

    '''
    # randomly initialize alpha
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    
    # forward pass interpolates through first and second part of discriminator network
    disc_interpolates = netD_D(netD(interpolates))#.view(batch_size,-1)
    
    # get gradients
    gradients, = autograd.grad(outputs=disc_interpolates.sum(), inputs=interpolates,
                              create_graph=True)
    
    # compute gradient penalty
    gradient_penalty = ((gradients.view(batch_size, -1).norm(2, dim=1) - 1) ** 
                        2).mean()* lamda
    return gradient_penalty

def get_matrix(netD, netD_Q, cluster_loader, label, dis_category):
    '''
    Computes confusion matrix using data in cluster_loader and label

    Parameters
    ----------
    netD : discriminator object
        module type of discriminator network.
    netD_Q : _netD_Q object
        module type of netD_Q auxiliary network.
    cluster_loader : DataLoader
        loader of data to generate predictions.
    label : array of int
        array with labels for data in cluster_loader.
    dis_category : int
        number of categories / clusters.

    Returns
    -------
    confusion_matrix : array of float 
        confusion matrix, true labels vs predicted labels. 
        Shape: (np.max(label)+1, dis_category).

    '''
    predict = []
    data_iter = iter(cluster_loader)
    
    for iteration in data_iter:
        # get image and true label from iterator 
        img, img_label = iteration
        # get predictions for each class using auxiliary network Q
        predict_label = netD_Q(netD(Variable(img.cuda())))
        predict.append(predict_label.data.cpu().numpy())  
        
    predict = np.concatenate(predict)
    predict_label = []
    
    # for all predictions in predict 
    for index in range(0, predict.shape[0]):
        # get max of predictions as predict_label
        predict_label.append(np.argmax(predict[index]))
        
    confusion_matrix = np.zeros((np.max(label)+1, dis_category), dtype=float)
    
    # update confusion matrix with predictions
    for index in range(0, len(predict)):
        confusion_matrix[label[index], predict_label[index]] += 1
        
    return confusion_matrix

def normalized(array):
    '''
    Normalize array

    Parameters
    ----------
    array : array of float

    Returns
    -------
    X : array of float
        Normalized array

    '''
    X = np.asarray([x.transpose((2, 0, 1)) for x in array])
    X = X.astype(np.float32)/(255.0 / 2) - 1.0
    return X

def rotation(array):
    '''
    Apply rotation on images from image array as augmentation

    Parameters
    ----------
    array : array of uint8
        array of images to rotate.

    Returns
    -------
    result : array of uint8
        Array of augmented images, 4 times larger than the original array.

    '''
    result_list = [array] 
    
    seq = iaa.Sequential([
        iaa.Flipud(1), # flip images vertically
    ])
    seq2= iaa.Sequential([
        iaa.Fliplr(1), # flip images horizontally
    ])
    seq3 = iaa.Sequential([
        iaa.Flipud(1), # flip images vertically
        iaa.Fliplr(1), # flip images horizontally
    ])
    
    # apply augmentations and add to result_list
    result_list.append(seq.augment_images(array))
    result_list.append(seq2.augment_images(array))
    result_list.append(seq3.augment_images(array))
    
    result = np.concatenate(result_list, axis=0)
    return result

def create_loader(array, shuffle=False, batchsize=1):
    '''
    Creates data loader from given array of images

    Parameters
    ----------
    array : array of float
        array of images.
    shuffle : bool, optional
        if True, data is reshuffled at every epoch. The default is False.
    batchsize : int, optional
        size of batch. The default is 1.

    Returns
    -------
    loader : DataLoader
        data loader from image array. Size: len(array)/batchsize

    ''' 
    label = torch.LongTensor(np.zeros((array.shape[0]), dtype=int))
    data = torch.FloatTensor(array)
    dataset = torch.utils.data.TensorDataset(data, label)
    
    loader = torch.utils.data.DataLoader(dataset, shuffle=shuffle, 
                                         batch_size=batchsize, num_workers=0)
    return loader

def create_model(rand=32, dis_category=5):
    '''
    Creates model, initialize generator, discriminator and auxiliary networks 

    Parameters
    ----------
    rand : int, optional
        number of noise variables. The default is 32.
    dis_category : int, optional
        number of categories / clusters. The default is 5.

    Returns
    -------
    netD : discriminator object
        module type of discriminator, first part of discriminator network.
    netG : generator object
        module type of generator network.
    netD_D : _netD_D object
        module type of netD_D, second part of discriminator network.
    netD_Q : _netD_Q object
        module type of netD_Q auxiliary network.

    '''
    netD, netG, netD_D, netD_Q = discriminator(), generator(
        rand = rand+dis_category), _netD_D(), _netD_Q(dis_category)
    
    # initialize weights to modules 
    netG.apply(weights_init)
    netD.apply(weights_init)
    netD_Q.apply(weights_init)
    netD_D.apply(weights_init)
    
    netD, netG, netD_D, netD_Q = netD.cuda(), netG.cuda(), netD_D.cuda(), netD_Q.cuda()
    
    return netD, netG, netD_D, netD_Q

def compute_purity_entropy(confusion_matrix):
    '''
    Compute purity and entropy metrics from confusion matrix 

    Parameters
    ----------
    confusion_matrix : array
        Confusion matrix from auxiliary network predictions.

    Returns
    -------
    entropy : float
        smaller entropy indicate better clustering.
    purity : float
        larger purity indicate better clustering.

    '''
    # predicted labels vs true labels
    clusters = np.transpose(confusion_matrix)
    clusters_none_zero = []
    entropy, purity = [], []
    
    # for each cluster / category of predicted labels 
    for cluster in clusters:
        cluster = np.array(cluster)
        
        # skip if no samples assigned ot this cluster 
        if cluster.sum() == 0.0:
            continue
        
        clusters_none_zero.append(cluster)
        
        # normalize cluster values 
        cluster = cluster / float(cluster.sum())
        
        # compute entropy and purity, add values to list 
        e = (cluster * [math.log((x + 1e-4), 2) for x in cluster]).sum()
        p = cluster.max()
        entropy += [e]
        purity	+= [p]
    
    # count of samples assigned to each non zero cluster 
    counts = np.array([c.sum() for c in clusters_none_zero])
    # normalize counts 
    coeffs = counts / float(counts.sum())
    
    # compute final entropy and purity
    entropy = -(coeffs * entropy).sum()
    purity = (coeffs * purity).sum()
    
    return entropy, purity

def get_f_score(confusion_matrix):
    '''
    Compute f-score from confusion matrix 

    Parameters
    ----------
    confusion_matrix : array
        Confusion matrix from auxiliary network predictions.

    Returns
    -------
    final_score : float
        final f-scoreN.

    '''
    final_score = .0
    fscore_list = []
    index_list = []
    TP_list = []
    
    # for each predicted cluster 
    for pred in range(0, confusion_matrix.shape[1]):
        index = np.argmax(confusion_matrix[:, pred])
        index_list.append(index)
    
    for n in range(0, 4):
        TP = .0
        precision_all = .0
        
        if n not in index_list:
            continue
        
        for m in range(0, len(index_list)):
            if index_list[m] == n:
                TP += confusion_matrix[n, m]
                precision_all += np.sum(confusion_matrix[:,m])
        
        if precision_all > 0:
            precision =  TP / precision_all
        else:
            precision = 0
        recall = TP/np.sum(confusion_matrix[n])
        
        if (recall+precision) > 0:
            fscore = (2 * recall * precision)/(recall + precision)
        else:
            fscore = 0
        
        TP_list.append(TP)
        fscore_list.append(fscore)
    
    TP_list = np.asarray(TP_list)
    for n in range(0, len(fscore_list)):
        final_score += fscore_list[n] * (TP_list[n]/np.sum(TP_list))
        
    return final_score

def get_category_matrix(loader , netD, netD_Q, dis_category):
    '''
    Generates list of samples per category / cluster for data in loader based on auxiliary network predictions

    Parameters
    ----------
    loader : DataLoader
        DESCRIPTION.
    netD : discriminator object
        module type of discriminator, first part of discriminator network.
    netD_Q : _netD_Q object
        module type of netD_Q auxiliary network.
    dis_category : int
        number of categories / clusters.

    Returns
    -------
    feature_dict : list
        list of number of samples per category / cluster. Size: dis_category

    '''
    test_iter = iter(loader)
    feature_dict = [0] * dis_category
    
    # for each sample in data loader
    for cluster_counting, data in enumerate(test_iter):
        # get category array from auxiliary network prediction
        category = netD_Q(netD(Variable(data[0].cuda()))).data.cpu().numpy()
        
        # update feature_dict with category
        for i in category:
            feature_dict[np.argmax(i)] += 1
            
    return feature_dict

def image_level_accuracy(positive_train_loader, positive_test_loader, negative_train_loader, negative_test_loader , netD, netD_Q, dis_category, experiment_root):

    proportion_1 = get_proportion(positive_train_loader , netD, netD_Q, dis_category)
    proportion_0 = get_proportion(negative_train_loader , netD, netD_Q, dis_category)
    proportion_test_1 = get_proportion(positive_test_loader , netD, netD_Q, dis_category)
    proportion_test_0 = get_proportion(negative_test_loader , netD, netD_Q, dis_category)

    estimator = KMeans(init='k-means++', n_clusters=2, n_init=1)
    true_label = [1]*proportion_1.shape[0] + [0]*proportion_0.shape[0]+[1]*proportion_test_1.shape[0] + [0]*proportion_test_0.shape[0]
    predict_label = estimator.fit_predict(np.concatenate([proportion_1,proportion_0,proportion_test_1,proportion_test_0],axis=0))
    predict_label = np.abs(1*(np.sum(np.array(true_label) == np.array(predict_label)) < np.sum(np.array(true_label) == (1- np.array(predict_label))))-predict_label)
    print('*k-means - f1_score:', f1_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                         predict_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], average='weighted'), 
          'recall:', recall_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                         predict_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], average='weighted'),
          'precision:', precision_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                         predict_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], average='weighted'))
                     
    #with open(experiment_root + "log","a") as f:
        #f.write('K_means_accuracy: ' +str(accuracy_all) + ' '+ str(accuracy_test) + '\n')
    with open(experiment_root + "log","a") as f:
        f.write('k-means - f1_score: '  + str(f1_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                         predict_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], average='weighted'))+ '\n' +  
          'recall: '+ str(recall_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                         predict_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], average='weighted'))+ '\n' + 
          'precision: '+ str(precision_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                         predict_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], average='weighted'))+ '\n')

    clf = svm.LinearSVC(penalty='l2')
    clf.fit(np.concatenate([proportion_1,proportion_0]), true_label[0: -proportion_test_1.shape[0]-proportion_test_0.shape[0]])
    print('SVM - f1_score:', f1_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                             predict_label, average='weighted'),
          'recall:', recall_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                             predict_label, average='weighted'),
          'precision:', precision_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                             predict_label, average='weighted'))
                     
    with open(experiment_root + "log","a") as f:
        f.write('SVM - f1_score:'+ str(f1_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                             predict_label, average='weighted'))+ '\n' + 
          'recall:'+ str(recall_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                             predict_label, average='weighted'))+ '\n' +
          'precision:'+ str(precision_score(true_label[-proportion_test_1.shape[0]-proportion_test_0.shape[0]:], 
                             predict_label, average='weighted'))+ '\n')

def get_proportion(loader_list, netD, netD_Q, dis_category):
    array = [get_category_matrix(i, netD, netD_Q, dis_category) for i in loader_list]
    array = np.asarray(array)
    array = array.astype(np.float32)
    
    for m, n in enumerate(array):
        k = n.astype(np.float32)/np.sum(n)
        array[m] = k
        
    return array

def train_representation(cell_train_set, cell_test_set, cell_test_label, 
                         positive_train_npy, positive_test_npy, negative_train_npy, 
                         negative_test_npy, netD, netG, netD_D, netD_Q, 
                         experiment_root, n_epoch=50, batchsize=32, rand=64, 
                         dis_category=5, ld = 1e-4, lg = 1e-4, lq = 1e-4,
                         save_model_steps=100, image_classification = False):
    '''
    Trains model of cell-level visual representation learning 
    One training iteration consists of 5 discriminator iterations, one generator iteration, and one auxiliary network iteration 
    Besides playing the minimax game between the generator (G) and the discriminator (D) through the EM distance, we also minimize the negative Log-likelihood between c and the output of the auxiliary network (Q(c|G(c, z)) to maximize mutual information

    Parameters
    ----------
    cell_train_set : array of uint8
        Array of training set images.
    cell_test_set : array of uint8
        Array of testing set images.
    cell_test_label : array of int
        Array of test labels.
    positive_train_npy : list
        list of positive training images.
    positive_test_npy : list
        list of positive testing images.
    negative_train_npy : list
        list of negative training images.
    negative_test_npy : list
        list of negative testing images.
    netD : discriminator object
        module type of discriminator, first part of discriminator network.
    netG : generator object
        module type of generator network.
    netD_D : _netD_D object
        module type of netD_D, second part of discriminator network.
    netD_Q : _netD_Q object
        module type of netD_Q auxiliary network.
    experiment_root : str
        root path of the experiment.
    n_epoch : int, optional
        number of epochs for training. The default is 50.
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
        if the training is for image classification or not. The default is False.

    Returns
    -------
    netD : discriminator object
        module type of discriminator, first part of discriminator network.
    netG : generator object
        module type of generator network.
    netD_D : _netD_D object
        module type of netD_D, second part of discriminator network.
    netD_Q : _netD_Q object
        module type of netD_Q auxiliary network.

    '''
    
    if not image_classification:
        # get augmented rotated images
        train = rotation(cell_train_set)
    
    # normalize sets and create loaders 
    train = normalized(train)
    train_loader = create_loader(train, shuffle=True, batchsize=batchsize)
    test = normalized(cell_test_set)
    test_loader = create_loader(test, shuffle=False, batchsize=1)
    test_label = cell_test_label

    if image_classification:
        positive_train_loader = [create_loader(
            normalized(n), shuffle=False, batchsize=64) for n in positive_train_npy]
        positive_test_loader = [create_loader(
            normalized(n), shuffle=False, batchsize=64) for n in positive_test_npy]
        negative_train_loader = [create_loader(
            normalized(n), shuffle=False, batchsize=64) for n in negative_train_npy]
        negative_test_loader =  [create_loader(
            normalized(n), shuffle=False, batchsize=64) for n in negative_test_npy]
    
    # define function to zero the gradients 
    def zero_grad():
        netD.zero_grad()
        netD_Q.zero_grad()
        netD_D.zero_grad()
        netG.zero_grad()
    
    # define Adam optimizers
    optimizerD = optim.Adam([
                {'params': netD.parameters(), 'lr': ld},
                {'params': netD_D.parameters(), 'lr': ld},
            ], betas=(0.5, 0.9))

    optimizerG = optim.Adam([
                {'params': netG.parameters(), 'lr': lg},
            ], betas=(0.5, 0.9))

    optimizerQ = optim.Adam([
                    {'params': netG.parameters()},
                    {'params': netD.parameters()},
                    {'params': netD_Q.parameters()},
                ], lq, betas=(0.5, 0.9))

    optimizerQ_G = optim.Adam([
                    {'params': netG.parameters()},            
                ], lg, betas=(0.5, 0.9))
    
    # initialize 
    input = torch.FloatTensor(batchsize, 3, 32, 32)
    noise = torch.FloatTensor(batchsize, rand+10, 1, 1)
    label = torch.FloatTensor(1)
    input, noise, label = input.cuda(), noise.cuda(), label.cuda()

    # initialize discrete variable c and gaussian variable z
    c = torch.randn(batchsize, 10)
    z = torch.randn(batchsize, rand)
    z, c = z.cuda(), c.cuda()
    
    # initialize criterions
    criterion = nn.BCELoss() # binary cross entropy
    criterion_logli = nn.NLLLoss() # negative log likelihood loss
    criterion_mse = nn.MSELoss() # mean squared error
    criterion, criterion_logli, criterion_mse = criterion.cuda(), criterion_logli.cuda(), criterion_mse.cuda()      
    
    # initialize
    gen_iterations = 0
    lamda = 10
    discrete_lamda = 1
    one = torch.tensor(1.0)
    mone = torch.tensor(-1.0)
    one, mone = one.cuda(), mone.cuda()
    fixed_noise = torch.from_numpy(fix_noise(dis_category=dis_category, 
                                             rand=rand)).cuda()
    values_D_G = []
    purities = []
    l_q = []
    best_purity = 0
    end = time.time()

    for epoch in range(n_epoch):
        
        print("current epoch:", epoch)

        dataiter = iter(train_loader)
        i = 0
        
        # len(train_loader) = len(train) / batchsize
        while i < len(train_loader):
            
            # we want to compute gradients with
            # respect to these parameters
            for p in netD.parameters(): 
                p.requires_grad = True 
            for p in netD_D.parameters(): 
                p.requires_grad = True 
            
            # one epoch consists of 5 discriminator iterations
            for iter_d in range(0, 5):
                
                if i >= len(train_loader):
                    continue

                zero_grad() # zero the gradients
                i += 1
                
                # get batch of images from dataiter
                # image_ shape: [batchsize, 3, 32, 32]
                image_, _ = dataiter.next()
                _batchsize = image_.size(0)
                image_ = image_.cuda()
                
                input.resize_as_(image_).copy_(image_)
                # wrap batch of input images tensor in variable
                inputv = Variable(input)

                # train with real
                # errD size = 1, mean errD for input batch
                errD_real = netD_D(netD(inputv)).mean()
                errD_real.backward(mone)

                # train with fake
                # draw samples from categorical dist
                rand_c, label_c = sample_c(_batchsize, dis_category=dis_category)
                rand_c = rand_c.cuda()
                # reshape 
                c.resize_as_(rand_c).copy_(rand_c)
                z.resize_(_batchsize, rand, 1, 1).normal_(0, 1)
                c = c.view(-1, dis_category, 1, 1)
                # get combined noise from c and z
                noise = torch.cat([c,z], 1)
                noise_resize = noise.view(_batchsize, rand+dis_category, 1, 1)
                
                # wrap noise tensor into variable
                if image_classification:
                    noisev = Variable(noise_resize)
                else:
                    noisev = Variable(noise_resize, requires_grad=True)
                
                # get fake inputv from generator network
                fake = Variable(netG(noisev).data)
                inputv = fake
                # errD size = 1, mean errD for input batch
                errD_fake = netD_D(netD(inputv)).mean()
                errD_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(netD_D, netD, input, 
                                                         fake.data, lamda, 
                                                         _batchsize)
                gradient_penalty.backward()
                
                # compute cost of the discriminator
                D_cost = -errD_real + errD_fake + gradient_penalty

                optimizerD.step()
  
            for p in netD.parameters(): 
                p.requires_grad = False 
            for p in netD_D.parameters(): 
                p.requires_grad = False 
            zero_grad()

            # draw samples from categorical dist
            rand_c,label_c = sample_c(batchsize,dis_category=dis_category)
            rand_c = rand_c.cuda()
            # reshape 
            c.resize_as_(rand_c).copy_(rand_c)
            z.resize_(batchsize, rand, 1, 1).normal_(0, 1)
            c = c.view(-1, dis_category, 1, 1)
            # get combined noise from c and z
            noise = torch.cat([c,z], 1)
            noise_resize = noise.view(batchsize, rand+dis_category, 1, 1)
            noisev = Variable(noise_resize)
            
            # one epoch consist of one generator iteration
            # get fake from generator network
            fake = netG(noisev)
            # get errG from discriminator network 
            errG = netD_D(netD(fake)).mean()
            errG.backward(mone)
            optimizerG.step()
            
            # get value function V(D, G)
            values_D_G.append((errD_real - errG).cpu().detach().numpy())
            # get purities
            confusion_matrix = get_matrix(netD, netD_Q, test_loader,
                                          test_label, dis_category)
            entropy, purity = compute_purity_entropy(confusion_matrix)
            f_score = get_f_score(confusion_matrix)
            purities.append(purity)

            for p in netD.parameters(): 
                p.requires_grad = True 
            for p in netD_D.parameters(): 
                p.requires_grad = True 
            zero_grad()
            
            # one epoch consist of one auxiliary network iteration
            # get inputv from combined noise from c and z
            inputv = Variable(noise_resize)
            # feed inputv to auxiliary network Q
            Q_c_given_x = netD_Q(netD(netG(inputv))).view(batchsize, dis_category)
            # the loss of the auxiliary network Q can be written as 
            # the negative log-likelihood between Q(c|G(c, z)) and 
            # the discrete variable c
            nll_loss = criterion_logli(Q_c_given_x, Variable(label_c.cuda()))
            mi_loss = discrete_lamda * nll_loss
            mi_loss.backward()
            optimizerQ.step()
            l_q.append(nll_loss)

            if gen_iterations % 10 == 0:

                batch_time = time.time() - end
                end = time.time()
                
                # write metrics to log file 
                with open(experiment_root + "log","a") as f:
                    f.write('batch_time:{0}, gen_iterations:{1}, D_cost:{2}, mi_loss:{3}'.format(batch_time/10, gen_iterations , -D_cost.data , mi_loss.data) + '\n')
                    
                # print metrics
                print('batch_time:{0}, gen_iterations:{1}, D_cost:{2}, mi_loss:{3}'.format(batch_time/10, gen_iterations , -D_cost.data , mi_loss.data))
                
                if purity > best_purity:
                    best_purity = purity
                    best_entropy = entropy
                    best_fscore = f_score
                    
                    # save best models
                    torch.save(netD.state_dict(), experiment_root + 
                           'model/netD_' + str(purity) + '_' + str(entropy)
                           + '_' + str(gen_iterations) + '.pth')
                    torch.save(netG.state_dict(), experiment_root + 
                               'model/netG_' + str(purity) + '_' + str(entropy)
                               + '_' + str(gen_iterations) + '.pth')
                    torch.save(netD_D.state_dict(), experiment_root +
                               'model/netD_D_' + str(purity) + '_' + str(entropy)
                               + '_' + str(gen_iterations) + '.pth')
                    torch.save(netD_Q.state_dict(), experiment_root +
                               'model/netD_Q_' + str(purity) + '_' + str(entropy)
                               + '_' + str(gen_iterations) + '.pth')


            if gen_iterations % 100 == 0:

                if image_classification:
                    # compute image level accuracy
                    image_level_accuracy(
                        positive_train_loader, positive_test_loader,
                        negative_train_loader, negative_test_loader, netD, netD_Q, 
                        dis_category, experiment_root)
                
                    end = time.time()
                
                # get sample of fake images from generator 
                G_sample = netG(Variable(fixed_noise))
                # save image 
                vutils.save_image(G_sample.data, 
                                  experiment_root + 'picture/fake_cell_' + 
                                  str(gen_iterations) + '.png', nrow=5, 
                                  normalize=True)
                
                # compute confusion matrix, entropy, purity and f-score
                confusion_matrix = get_matrix(netD, netD_Q, test_loader,
                                              test_label, dis_category)
                entropy, purity = compute_purity_entropy(confusion_matrix)
                f_score = get_f_score(confusion_matrix)
                print('purity:', purity, 'entropy:', entropy, 'f_score', f_score)
                
                # write to log file 
                with open(experiment_root + "log","a") as f:
                    f.write('gen_iterations: ' + str(gen_iterations) + 'purity: ' + 
                            str(purity) + ' entropy: ' + str(entropy) + ' f_score: ' 
                            + str(f_score)+ '\n')
            
            # save models 
            if gen_iterations % save_model_steps == 0 :
                '''
                torch.save(netD.state_dict(), experiment_root + 
                           'model/netD_' + str(purity) + '_' + str(entropy)
                           + '_' + str(gen_iterations) + '.pth')
                torch.save(netG.state_dict(), experiment_root + 
                           'model/netG_' + str(purity) + '_' + str(entropy)
                           + '_' + str(gen_iterations) + '.pth')
                torch.save(netD_D.state_dict(), experiment_root +
                           'model/netD_D_' + str(purity) + '_' + str(entropy)
                           + '_' + str(gen_iterations) + '.pth')
                torch.save(netD_Q.state_dict(), experiment_root +
                           'model/netD_Q_' + str(purity) + '_' + str(entropy)
                           + '_' + str(gen_iterations) + '.pth')
                '''
                end = time.time()
                
            gen_iterations += 1
            
    # compute confusion matrix, entropy, purity and f-score
    confusion_matrix = get_matrix(netD, netD_Q, test_loader,
                                  test_label, dis_category)
    entropy, purity = compute_purity_entropy(confusion_matrix)
    f_score = get_f_score(confusion_matrix)
    print('purity:', purity, 'entropy:', entropy, 'f_score', f_score)
    print('best purity:', best_purity, 'best entropy:', best_entropy, 'best f_score:', best_fscore)

    return values_D_G, l_q, purities
