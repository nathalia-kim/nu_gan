from gan_model import create_model, create_loader, normalized
import torch 
from torch.autograd import Variable
import torch.nn as nn
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import numpy as np


def cell_classification(experiment_id, experiment_root, X_train_path, X_test_path, y_train_path, y_test_path, max_iter=3000, rand=32, dis_category=5, netD_fn="netD_best_model.pth"):
    '''
    Performs cell classification using trained model as feature extractor and L2-SVM as classifier

    Parameters
    ----------
    experiment_id : int
        id of experiment for using pretrained model.
    experiment_root : str
        root path of the experiment.
    X_train_path : str
        path of the train data.
    X_test_path : str
        path of the test data.
    y_train_path : str
        path of the train labels.
    y_test_path : str
        path of the test labels.
    netD_fn : str
        filename of the pth file with pretrained weights.
    max_iter: int
        number of max iterations to train classifier 
    rand : int, optional
        number of gaussian noise variables. The default is 32.
    dis_category : int, optional
        number of categories / clusters. The default is 5.

    Returns
    -------
    None.

    '''
     
    model_path = experiment_root + "/" + str(experiment_id) + "/model/"
        
    # instantiate models 
    netD, netG, netD_D, netD_Q = create_model(rand=rand, dis_category=dis_category)
    
    # load model from pth file 
    netD.load_state_dict(torch.load(model_path + netD_fn))
    
    # load datasets
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    # concatenate datasets for preping, split later 
    data = np.concatenate([X_train, X_test])
    data_labels = np.concatenate([y_train, y_test])
    
    # get data loader 
    data = normalized(data)
    data_loader = create_loader(data, shuffle=False, batchsize=len(data))
    
    dataiter = iter(data_loader)
    
    # iterate through images 
    images, labels = dataiter.next()
    images = images.cuda()
    # wrap batch of input images tensor in variable
    inputv = Variable(images)
    
    # get features after each residual block 
    outputs= []
    def hook(module, input, output):
        outputs.append(output)
    handle1 = netD.layer_down_1.register_forward_hook(hook)
    handle2 = netD.layer_down_2.register_forward_hook(hook)
    handle3 = netD.layer_none_1.register_forward_hook(hook)
    handle4 = netD.layer_none_2.register_forward_hook(hook)
    
    # forward pass
    out = netD(inputv)
    
    # max pooling, downsampling to a 4x4 grid 
    m1 = nn.MaxPool2d(4)
    out1 = m1(outputs[0])
    m2 = nn.MaxPool2d(2)
    out2 = m2(outputs[1])
    out3 = m2(outputs[2])
    out4 = m2(outputs[3])
    
    # concatenate features 
    features = torch.cat((out1, out2, out3, out4), 1)
    
    # flatten features 
    features = features.cpu().detach().numpy()
    features = features.reshape((len(data), 
                                 features.shape[1]*features.shape[2]*features.shape[3]))
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)
        
    print("Extracting features done")
    
    # split data 
    X_train, X_test, y_train, y_test = train_test_split(features, data_labels, test_size=0.2, random_state=42)
    
    # create classifier
    clf = svm.LinearSVC(C=1.0, penalty="l2", max_iter=max_iter, dual=False)
    
    print("Fitting the model...")
    clf.fit(X_train, y_train)
    print("Done")
    
    # predict
    yhat_test = clf.predict(X_test)
    
    print('SVM testing - f1_score:', f1_score(y_test, yhat_test, average='weighted'),
              'recall:', recall_score(y_test, yhat_test, average='weighted'),
              'precision:', precision_score(y_test, yhat_test, average='weighted'))
    
    # remove hooks
    handle1.remove()
    handle2.remove()
    handle3.remove()
    handle4.remove()