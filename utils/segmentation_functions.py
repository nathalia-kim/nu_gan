import os
import cv2
import numpy as np
import histomicstk as htk
import scipy as sp
import skimage.io
import skimage.measure
import skimage.color
import matplotlib.pyplot as plt
from PIL import Image
import glob
import math
plt.rcParams['figure.figsize'] = 15, 15
plt.rcParams['image.cmap'] = 'gray'
titlesize = 24


def save_normalized_images(inputImageFile, refImageFile, save_path):
    '''
    Performs stain normalization and saves resulting image 

    Parameters
    ----------
    inputImageFile : str
        path to input image.
    refImageFile : str
        path to reference image.
    save_path : str
        path to save resulting image.

    Returns
    -------
    image : array
        array of resulting normalized image.

    '''
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    name = inputImageFile.split('/')[-1].split('.')[0]
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)
    skimage.io.imsave(save_path + name + '.png', imNmzd)
    image = Image.open(save_path + name + '.png')
    return image


def cell_segment_evaluate(intensity, refImageFile, segmenval_original_path, segmenval_label_path):
    '''
    Evaluate cell segmentation, compute f-score, precision and recall

    Parameters
    ----------
    intensity : int
        intensity fior image thresholding.
    refImageFile : str
        path to reference image.
    segmenval_original_path : str
        path to original images.
    segmenval_label_path : str
        path to image labels.

    Returns
    -------
    None.

    '''
    totallabel  =0
    totalsegment  = 0
    totalright  = 0
    root_dir = segmenval_original_path
    imList = os.listdir(root_dir)
    for imdata in range(0,len(imList)):

        inputImageFile = (segmenval_original_path + imList[imdata])

        name =  imList[imdata].strip('.png')
        imInput = skimage.io.imread(inputImageFile)[:, :, :3]
        imReference = skimage.io.imread(refImageFile)[:, :, :3]
        # get mean and stddev of reference image in lab space
        meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
        # perform reinhard color normalization
        imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)

        w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imNmzd,I_0=255 )
        I_0=255
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        # specify stains of input image
        stains = ['hematoxylin',  # nuclei stain
                  'eosin',        # cytoplasm stain
                  'null']    
        # Perform color deconvolution
        deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)

        imNucleiStain = deconv_result.Stains[:, :, 1]

        foreground_threshold = intensity

        imFgndMask = sp.ndimage.morphology.binary_fill_holes(
            imNucleiStain < foreground_threshold)

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
        # compute nuclei properties
        objProps = skimage.measure.regionprops(imNucleiSegMask)
       # print 'Number of nuclei = ', len(objProps)

        # prepare groundtruth
        root_data= segmenval_label_path + name+'/'
        imageList = os.listdir(root_data)

        labelist = []

        for i in imageList[0:len(imageList)]:
            img = cv2.imread(root_data + i)
            img = 255-img
            im = img[:,:,1]
            im.reshape(200,200)
            labelist.append(im)

        imNucleicompact = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)

        k= (imNucleicompact==-1)
        imNucleicompact1=np.copy(k)
        plt.rcParams['figure.figsize'] = 15,15    
        for ii in range(0,k.shape[0]):
            for jj in range(0,k.shape[1]):
                if imNucleicompact[ii,jj]>0:
                    imNucleicompact1[ii,jj]=1

        imNucleicompact2 = skimage.measure.label(imNucleicompact1,connectivity = 1)

        right = 0
        segment = 0
        label = 0
        for i in range(1,imNucleicompact2.max()):

            k =  (imNucleicompact2==i)
            location = np.where(k == 1)
            x_min, y_min = min(location[0]),min(location[1])
            x_max, y_max = max(location[0]),max(location[1])
            space = (x_max-x_min)*(y_max-y_min)

            if space<450 and space>100:

                for im in labelist:
                    result = k*im
                    if  result.sum()>255*100:
                        right= result.sum()/255 + right
                        segment = k.sum() + segment

            if space>449:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7)) 
                k.dtype=np.uint8
                eroded=cv2.erode(k,kernel);
                dilated = cv2.dilate(eroded,kernel)
                new_seg = skimage.measure.label(dilated,connectivity = 1)
                for j in range (1,new_seg.max()+1):

                    kk =  (new_seg==j)
                    location1 = np.where(kk == 1)
                    x_min1, y_min1 = min(location1[0]),min(location1[1])
                    x_max1, y_max1 = max(location1[0]),max(location1[1])
                    space1 = (x_max1-x_min1)*(y_max1-y_min1)
                    if space1< 800:
                        for im in labelist:
                            result = kk*im
                            if  result.sum()>255*100:
                                right= result.sum()/255 + right
                                segment = kk.sum() + segment
                                
        # calculate the number of pixel in ground truth, segmentation result and overlapping region
        label= 0
        for im in labelist:
            label = label+ im.sum()/255

        totallabel  =label+totallabel
        totalsegment  = segment+totalsegment
        totalright  = right+totalright

    a=totallabel
    b=totalsegment
    c=totalright
    
    # calculate f-score
    recall = c/a
    precision = c/float(b)
    Fscore=(2*precision*recall)/(precision+recall)
    print('recall, precision:')
    print(recall, precision)
    print('Fscore:')
    print(Fscore)

def masks_to_npy(images_path, ref_path, output_path):
    '''
    Generate npy file with segmented image from binary masks

    Parameters
    ----------
    images_path : str
        path to images.
    ref_path : str
        path to reference image.
    output_path : str
        path to save the output npy file.

    Returns
    -------
    None.

    '''
    imList = []
    ids = []
    
    # get list of ids for all files
    for image_path in glob.glob(images_path + "*"):
        # get image id
        id = os.path.basename(image_path)[0:os.path.basename(image_path).find("_")]
        ids.append(id)
    
    # get list of unique ids
    ids = np.array(ids)
    img_ids = np.unique(ids)
    
    for img_id in img_ids:
        # read image
        image = skimage.io.imread(images_path + img_id + "_crop.png")
        # read mask 
        mask = skimage.io.imread(images_path + img_id +  "_labeled_mask_corrected.png")
    
        # apply stain normalization on image 
        # read reference image 
        imRef_path = glob.glob(ref_path + "*")
        imRef = skimage.io.imread(imRef_path[0])
        # get mean and stddev of reference image in lab space
        meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imRef)
        # perform reinhard color normalization
        imNorm = htk.preprocessing.color_normalization.reinhard(image, meanRef, stdRef)       
    
        # loop through labels in mask, skip label 0 (background)
        for i in range(1, max(np.unique(mask))+1):
            single_cell = imNorm.copy()
            single_cell[mask != i] = [0, 0, 0]
        
            # convert to grayscale
            gray = cv2.cvtColor(single_cell, cv2.COLOR_RGB2GRAY) 
        
            # threshold to get just the signature 
            retval, thresh_gray = cv2.threshold(gray, thresh=10, maxval=255, \
                                           type=cv2.THRESH_BINARY_INV)
        
            # get contours
            cnts, im = cv2.findContours(thresh_gray,cv2.RETR_LIST, \
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            # get bounding box from contours
            single_cell[mask != i] = [255, 255, 255]
            x,y,w,h = cv2.boundingRect(cnts[0])
        
            # crop object around bounding box
            crop = single_cell[y:y+h, x:x+w]
        
            # resize singe-cell images to 32x32x3
            resized = crop.copy()
            height, width = crop.shape[0], crop.shape[1]
            if max(height, width) > 32:
                scale = 32/float(max(height,width))
                height, width = int(height*scale), int(width*scale)
                resized = np.array(Image.fromarray(crop).resize((width, height)))
            
            height, width = resized.shape[0], resized.shape[1]
            if min(height, width) < 32:
                v_pad = 32-height
                h_pad = 32-width
                resized = cv2.copyMakeBorder(resized, math.floor(v_pad/2), math.ceil(v_pad/2), math.floor(h_pad/2), math.ceil(h_pad/2), cv2.BORDER_CONSTANT, value=(255,255,255))
            
            # add single-cell image to list
            imList.append(resized)
    
    # save image list to npy file 
    imList = np.array(imList)
    # save npy
    np.save(output_path + 'Train.npy', imList)
    

def cell_segment(image_path, data_saved_path, ref_path, intensity):
    '''
    Perform cell segmentation on images

    Parameters
    ----------
    image_path : str
        path with images.
    data_saved_path : str
        path to save the result.
    ref_path : str
        path with reference image.
    intensity : int
        intensity for image thresholding.

    Returns
    -------
    None.

    '''

    plt.rcParams['figure.figsize'] = 15, 15
    plt.rcParams['image.cmap'] = 'gray'
    
    # get image id / name 
    name = image_path.split('/')[-1].split('/')[-1].split('.')[0]
    
    # read current image 
    inputImageFile = image_path
    imInput = skimage.io.imread(inputImageFile)[:, :, :3]
    
    # read reference image 
    refImageFile = ref_path
    imReference = skimage.io.imread(refImageFile)[:, :, :3]
    
    # get mean and stddev of reference image in lab space
    meanRef, stdRef = htk.preprocessing.color_conversion.lab_mean_std(imReference)
    
    # perform reinhard color normalization
    imNmzd = htk.preprocessing.color_normalization.reinhard(imInput, meanRef, stdRef)

    # Perform color deconvolution
    w_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(imNmzd, I_0=255 )
    I_0 = 255    
    deconv_result = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, w_est, I_0)

    imNucleiStain = deconv_result.Stains[:, :, 1]
    
    # binary thresholding
    foreground_threshold = intensity
    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < foreground_threshold)

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
    # compute nuclei properties
    #objProps = skimage.measure.regionprops(imNucleiSegMask)
    print ('step one done')

    imNucleicompact = htk.segmentation.label.compact(imNucleiSegMask, compaction=3)
    k= (imNucleicompact==-1)
    imNucleicompact1=np.copy(k)
    plt.rcParams['figure.figsize'] = 15,15    
    for ii in range(0,imNucleicompact.shape[0]):
        for jj in range(0,imNucleicompact.shape[1]):
            if imNucleicompact[ii,jj]>0:
                imNucleicompact1[ii,jj]=1

    imNucleicompact2 = skimage.measure.label(imNucleicompact1,connectivity = 1)

    plt.rcParams['figure.figsize'] = 1, 1

    # save image and calculate f-score 
    listt = []
    seglis = []
    list_nuclei = []

    for i in range(1,imNucleicompact2.max()):

        k =  (imNucleicompact2==i)
        location = np.where(k == 1)
        x_min, y_min = min(location[0]),min(location[1])
        x_max, y_max = max(location[0]),max(location[1])
        space = (x_max-x_min)*(y_max-y_min)

        if space<450 and space>100:           
            segmentate = k[x_min:x_max,y_min:y_max]
            segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
            listt.append([x_min,y_min,x_max,y_max])
            seglis.append(segmentate)
            img1 = imNmzd[x_min:x_max,y_min:y_max,:]
            img1 = img1*segmentate
            list_nuclei.append(img1)
            #plt.imshow(img1)
            #plt.show()

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
                location1 = np.where(kk == 1)
                x_min1, y_min1 = min(location1[0]),min(location1[1])
                x_max1, y_max1 = max(location1[0]),max(location1[1])
                space1 = (x_max1-x_min1)*(y_max1-y_min1)
                if space1< 800:
                    segmentate = kk[x_min1:x_max1,y_min1:y_max1]
                    segmentate = np.tile(np.expand_dims(segmentate,axis=2),(1,1,3))
                    listt.append([x_min1,y_min1,x_max1,y_max1])
                    seglis.append(segmentate)
                    img1 = imNmzd[x_min1:x_max1,y_min1:y_max1,:]
                    img1 = img1*segmentate
                    list_nuclei.append(img1)


    #save cell image filled with [255,255,255]
    image_dict = {}
    n=0
    #preparing cell images
    for img in list_nuclei:
        color_mean = img.mean(axis=2)
        for p in range(0, color_mean.shape[0]):
            for q in range(0, color_mean.shape[1]):
                if color_mean[p, q] == 0.0:
                    img[p, q, :] = 255

        height, width = img.shape[0], img.shape[1]
        if max(height,width)>32:
            scale = 32/float(max(height,width))
            height, width = int(height*scale), int(width*scale)
            #img = sp.misc.imresize(img, (height, width))
            img = np.array(Image.fromarray(img).resize((width, height)))

        npad = ((int(16-height/2),int(32-height-(16-height/2))),(int(16-width/2),int(32-width-(16-width/2))),(0,0))
        segmentate_image = np.pad(img, pad_width=npad,constant_values=255,mode='constant')
        image_dict[n] = segmentate_image
        n+=1

    image = np.array(list(image_dict.values()))
    np.save(((data_saved_path + name)+'.npy'), image)
    print ('Number of nuclei = ', len(image_dict))
    print ('image saved')