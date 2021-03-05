import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from random import randrange
import pandas as pd

import numpy as np
import histomicstk as htk
import numpy as np
import scipy as sp
import skimage.io
import skimage.measure
from skimage.measure import label
import skimage.color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import ctypes
import imutils

experiment_root = '/Users/kim/experiment/'
output_path = '/Users/kim/Documents/GitHub/nu_gan/figures/'


positive_images_root = experiment_root + 'data/original/positive_images/' 
negative_images_root = experiment_root + 'data/original/negative_images/' 
positive_npy_root = experiment_root + 'data/segmented/positive_npy/'
negative_npy_root = experiment_root + 'data/segmented/negative_npy/'
ref_path = experiment_root + 'data/original/reference/BM_GRAZ_HE_0007_01.png'
intensity = 160

# Figure 2: Positive and negative examples of bone marrow images
def figure_2(positive_images_root, negative_images_root):
    # get positive and negative images from root 
    positives = glob.glob(positive_images_root + "*")
    negatives = glob.glob(negative_images_root + "*")
    
    # get random image
    rand_positive = positives[randrange(len(positives))]
    rand_negative = negatives[randrange(len(negatives))]
    
    # read images 
    positive = cv2.imread(rand_positive)
    positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
    negative = cv2.imread(rand_negative)
    negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')

    fig.suptitle('Figure 2')
    ax1.imshow(positive)
    ax1.set_title('Positive')
    ax2.imshow(negative)
    ax2.set_title('Negative')

# Figure 6: Overview of the segmentation process
# code mostly from segmentation_functions.cell_segment
def figure_6(positive_images_root, ref_path):
    # take positives as an example
    positives = glob.glob(positive_images_root + "*")
    # get random image
    rand_positive = positives[randrange(len(positives))]
    
    # get image id / name 
    name = rand_positive.split('/')[-1].split('/')[-1].split('.')[0]
    
    # create output directory
    os.makedirs(output_path+str(name)+'/')
    os.chdir(output_path+str(name)+'/')

    # read current image 
    inputImageFile = rand_positive
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]    
    plt.imsave("original_image.png", imInput)
    
    # read reference image 
    refImageFile = ref_path
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    
    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)

    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
    plt.imsave("normalized_image.png", imNmzd)
    
    # Perform color deconvolution
    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imNmzd, I_0=255 )
    I_0 = 255
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
              'eosin',        # cytoplasm stain
              'null']    
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)
    # get only the nuclei / hematoxylin stain
    imNucleiStain = deconv_result.Stains[:, :, 1]
    plt.imsave("nuclei_stain.png", imNucleiStain)
    
    # binary thresholding
    foreground_threshold = intensity
    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < foreground_threshold)
    plt.imsave("binary_thresholding.png", imFgndMask)
    
    min_radius = 5
    max_radius = 30
    imLog = htk.filters.shape.clog(imNucleiStain, imFgndMask,
                                   sigma_min=min_radius * np.sqrt(2),
                                   sigma_max=max_radius * np.sqrt(2))
    
    # detect and segment nuclei using local maximum clustering
    local_max_search_radius = 10
 
    imNucleiSegMask1, Seeds, Max = htk.segmentation.nuclear.max_clustering(
        imLog[0], imFgndMask, local_max_search_radius)
    # filter out small objects
    min_nucleus_area = 200
    imNucleiSegMask = htk.segmentation.label.area_open(
        imNucleiSegMask1, min_nucleus_area).astype(np.int)
    plt.imsave("labeled_image.png", imNucleiSegMask)
    
    # label to rgb
    #labeled_img = skimage.color.label2rgb(imNucleiSegMask, bg_label = 0) 
    label_image = skimage.measure.label(imNucleiSegMask)
    image_label_overlay = skimage.color.label2rgb(label_image, image=imInput, bg_label=0)
    plt.imsave("labeled_image_rgb.png", image_label_overlay)   
    
    imNucleicompact = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)
    k= (imNucleicompact==-1)
    imNucleicompact1=np.copy(k)
    plt.rcParams['figure.figsize'] = 15,15    
    for ii in range(0,imNucleicompact.shape[0]):
        for jj in range(0,imNucleicompact.shape[1]):
            if imNucleicompact[ii,jj]>0:
                imNucleicompact1[ii,jj]=1

    imNucleicompact2 = skimage.measure.label(imNucleicompact1,connectivity = 1)
    imInput2 = np.copy(imNmzd)
    plt.rcParams['figure.figsize'] = 1, 1

    # save image and calculate f-score #########
    listt = []
    seglis = []
    list_nuclei = []
    right = 0
    segment = 0
    label = 0
    image = imInput.copy()
    
    for i in range(1,imNucleicompact2.max()):

        k =  (imNucleicompact2==i)
        location = np.where(k == 1)
        x_min, y_min = min(location[0]),min(location[1])
        x_max, y_max = max(location[0]),max(location[1])
        space = (x_max-x_min)*(y_max-y_min)

        if space<450 and space>100:           
            # find countour around k
            mask = np.zeros(k.shape, dtype="uint8")
            mask[k == True] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # draw contour on image 
            cv2.drawContours(image, cnts, -1, (0,255,0), 1)

        if space>449:
            #print i
            #print space
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7)) 
            k.dtype=np.uint8
            eroded=cv2.erode(k,kernel);
            dilated = cv2.dilate(eroded,kernel)
            new_seg = skimage.measure.label(dilated,connectivity = 1)
            for j in range (1,new_seg.max()+1):
                #print 'j=',j
                kk =  (new_seg==j)
            # find countour around k
            mask = np.zeros(k.shape, dtype="uint8")
            mask[kk == True] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # draw contour on image 
            cv2.drawContours(image, cnts, -1, (0,255,0), 1)    
            
    plt.imsave("final_segmentation.png", image)
    
    

# show segmentation results 
def figure_segment_results(positive_npy_root, negative_npy_root):
    # get positive and negative images from root 
    positives = glob.glob(positive_npy_root+str(intensity)+'/' + "*")
    negatives = glob.glob(negative_npy_root+str(intensity)+'/' + "*")
    
    # read first npy files 
    positive = np.load(positives[1], allow_pickle=True)
    negative = np.load(negatives[2], allow_pickle=True)  
    
    # get 5 positive and negative samples
    positives_samples = (positive[0:5])
    negatives_samples = (negative[0:5])
    # append samples together 
    samples = np.append(positives_samples, negatives_samples, axis = 0)
    
    # create plot with samples 
    fig, ax = plt.subplots(2,5, figsize=(20,10))
    for i in range(10):
        img = samples[i]
        ax[i//5, i%5].imshow(img)
        if i < 5:
            ax[i//5, i%5].set_title("Positives")
        else:
            ax[i//5, i%5].set_title("Negatives")
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_aspect('auto')
    plt.show()

# plot metrics for representation learning 
def plot_representation(experiment_number):
    file_path = experiment_root + str(experiment_number) + "/log"
    
    log_data  = open(file_path, 'r')
    split_list = []
    
    for line in log_data:
        # get list of values
        values = line.split(", ")
        
        # get separate values 
        batch_time = values[0].split(":")[1]
        gen_iterations = values[1].split(":")[1]
        D_cost = values[2].split(":")[1]
        mi_loss = values[3].split(":")[1]
            
        split_list.append([float(batch_time), float(gen_iterations), float(D_cost), float(mi_loss)])
    
    df = pd.DataFrame(split_list, columns=['batch_time', 'gen_iterations', 'D_cost', 'mi_loss'])
    
    # plot loss over gen_iterations
    df.plot(x="gen_iterations", y="mi_loss", figsize=(6, 4), xlabel="Generator iterations", ylabel="Loss")
    
    # get purity and entropy from model files
    models_path = experiment_root + str(experiment_number) + "/model/"
    metrics = []
    
    for file in glob.glob(models_path + "*"):
        filename = os.path.basename(file)
        
        if "netG" in filename:
            purity = filename.split("_")[1]
            entropy = filename.split("_")[2]
            iteration = filename.split("_")[3]
            iteration = iteration.replace(".pth",'')
            
            metrics.append([float(purity), float(entropy), float(iteration)])
            
    metrics = pd.DataFrame(metrics, columns=["purity", "entropy", "iteration"])
    metrics = metrics.sort_values(by=['iteration'])
    
    metrics.plot(x="iteration", y = "purity", figsize=(6, 4), xlabel="Generator iterations", ylabel="Purity")
    metrics.plot(x="iteration", y = "entropy", figsize=(6, 4), xlabel="Generator iterations", ylabel="Entropy")
    
            
        
    

    