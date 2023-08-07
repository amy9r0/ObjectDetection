#!/usr/bin/env python
# coding: utf-8

# In[19]:
import pandas as pd
import numpy as np

def AP(output, ground_truth):
    """
        input: output: from detection model
                        list of dictionaries: size of number_images X 1
                        dictionary keys: boxes, labels, scores
                            boxes are xmin, ymin, xmax, ymax
                ground_truth: list of dictionaries: size of number_images x 1
                        dictionary keys: boxes, labels
        returns: AP: average precision on a class
                        dictionary of classes as keys
                mAP: average precision across all classes
    """
    threshold = 0.5
    # collecting ground truth labels
    categories = get_labels(ground_truth)
    
    # setup dataframe of lists to cumulate counts 
    cumulation = create_dataframe(categories, ground_truth)

    # looking at each image
    for k, out in enumerate(output):
        # relabeling for ease of reading
        true_boxes = ground_truth[k]['boxes'].cpu().numpy()
        true_labels = ground_truth[k]['labels'].cpu().numpy()
        bboxes = out['boxes'].cpu().numpy()
        labels = out['labels'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        # setting a true positive flag for each ground truth box
        TP_flag = [0]*len(true_labels)
        
        for i, box in enumerate(bboxes):
            # iterating over predicted boxes
            # checking predicted labels are in tracked truth labels
            # cannot compute AP and mAP for labels not in truth set
            if labels[i] not in categories:
                continue
            
            for j, g_box in enumerate(true_boxes):
                # iterating over each ground_truth box
                    
                if TP_flag[j] or (labels[i] != true_labels[j]):
                    # checking for true positive already established
                    # checking if predicted label doesn't match true
                    # updating FP for prediction category
                    
                    update_FP_count(cumulation, labels[i])
                    continue
                    
                dim_intersect = get_intersect_dim(g_box, box)
                
                if len(dim_intersect) < 1:
                    # checking for overlap
                    # updating frame with FP
                    
                    update_FP_count(cumulation, labels[i])
                    continue
                    
                # calculating IoU
                IoU = get_IoU(g_box, box, dim_intersect)
                
                if IoU < threshold:
                    # FP update if IoU too small
                    
                    update_FP_count(cumulation, labels[i])
                    continue
                
                # updating frame with TP
                # updating TP flag
                update_TP_count(cumulation, labels[i])
                TP_flag = 1
         
    # calculate the AP for each category
    AP = {}
    for cat in categories:
        AP[cat] = get_AP(cumulation[str(cat)])
        
    # calculate mAP
    mAP = 0
    for cat in AP:
        mAP = mAP+AP[cat]
    mAP = mAP/len(categories)
    
    return mAP, AP


# In[20]:


def update_FP_count(frame, label):
    # updating metrics for false positive result
    # frame: dataframe of categories and list of metrics
    # label is category for result
    # complicated by possible initialization of lists
    
    TP = frame[str(label)]['TP']
    FP = frame[str(label)]['FP']
    precision = frame[str(label)]['precision']
    recall = frame[str(label)]['recall']
    
    if not TP:
        frame[str(label)]['TP'] = [0]
        frame[str(label)]['FP'] = [1]
        frame[str(label)]['precision'] = [0]
        frame[str(label)]['recall'] = [0]
    else:
        frame[str(label)]['TP'] = TP.append(TP[-1])
        frame[str(label)]['FP'] = FP.append(FP[-1]+1)
        frame[str(label)]['precision'] = precision.append(TP[-1]/(TP[-1]+FP[-1]+1))
        frame[str(label)]['recall'] = recall.append(TP[-1]/frame[str(label)]['ground_count'])


# In[21]:


def update_TP_count(frame, label):
    # updating metrics for true positive result
    # frame: dataframe of categories and list of metrics
    # label is category for result
    # complicated by possible initialization of list
 
    TP = frame[str(label)]['TP']
    FP = frame[str(label)]['FP']
    precision = frame[str(label)]['precision']
    recall = frame[str(label)]['recall']
    
    if not TP:
        frame[str(label)]['TP'] = [1]
        frame[str(label)]['FP'] = [0]
        frame[str(label)]['precision'] = [1]
        frame[str(label)]['recall'] = 1/frame[str(label)]['ground_count']
    else:
        frame[str(label)]['TP'] = TP.append(TP[-1]+1)
        frame[str(label)]['FP'] = FP.append(FP[-1])
        frame[str(label)]['precision'] = precision.append((TP[-1]+1)/(TP[-1]+FP[-1]+1))
        frame[str(label)]['recall'] = recall.append((TP[-1]+1)/frame[str(label)]['ground_count'])



# In[22]:


def get_area(bbox):
    # assumes [xmin, ymin, xmax, ymax]
    return (bbox[3]-bbox[1]) * (bbox[2]-bbox[0])


# In[23]:


def get_intersect_dim(true_box, pred_box):
    dim_intersect = list()
    # checking each dimension individually
    for coord in [0, 1]:
        # defining our corners in one dimension
        small_max = min(true_box[coord+2], pred_box[coord+2])
        large_max = max(true_box[coord+2], pred_box[coord+2])
        small_min = min(true_box[coord], pred_box[coord])
        large_min = max(true_box[coord], pred_box[coord])
                    
        if not (small_max < large_min):
            # checking for overlap and setting length
            dim_intersect.append(small_max - large_min)
                        
        # resetting overlap dim if either does not exist
        if len(dim_intersect) < 2:
            dim_intersect = []
    return dim_intersect


# In[24]:


def get_IoU(true_box, pred_box,i_dims):
    # true_box: ground truth box [xmin, ymin, xmax, ymax]
    # pred_box: predicted box [xmin, ymin, xmax, ymax]
    # i_dims: two dimensional list of intersect
    # returns area of intersection over union
    area_i = i_dims[0]*i_dims[1]
    area_ground = get_area(true_box)
    area_pred = get_area(pred_box)
    IoU = area_i/(area_ground+area_pred-area_i)
    return IoU


# In[25]:


def get_AP(category):
    # category: series of lists with labels: precision, recall
    # returns average precision: area under recall/precision curve
    
    precision = category['precision']
    recall = category['recall']
    
    # recall should always be highest at end of string
    # true positive count only increases and ground truth count is constant
    precision.reverse()
    recall.reverse()
    
    AP = 0
    high_prec = 0
    for i in np.arange(1, len(recall)):
        # defining width of rectangle for every recall step
        width = (recall[i-1]-recall[i])
        if precision[i] > precision[i-1]:
            AP = AP + width*precision[i-1] + width*(precision[i]-precision[i-1])/2
        elif precision[i] < precision[i-1]:
            AP = AP + width*precision[i] + width*(precision[i-1]-precision[i])/2
            
    return AP


# In[28]:


def create_dataframe(categories, truth):
    # categories: list of true image labels
    # truth: ground truth information
    # returns dataframe

    cumulation = pd.DataFrame(columns = categories, index = ['TP','FP','precision', 'recall','ground_count'])
    for cat in categories:
        cumulation[cat]['TP'] = []
        cumulation[cat]['FP'] = []
        cumulation[cat]['precision'] = []
        cumulation[cat]['recall'] = []
        cumulation[cat]['ground_count'] = 0
        for img in truth:
            # getting ground truth count for all categories
            for label in img['labels'].cpu():
                if str(label.item()) == cat:
                    cumulation[cat]['ground_count'] = cumulation[cat]['ground_count'] + 1
    
    return cumulation


# In[27]:


def get_labels(ground_truth):
    # truth: ground truth information
    # output: output information from detection model
    # returns list of categories
    
    categories = list()
    for truth in ground_truth:
        for label in truth['labels']:
            if str(label.item()) not in categories:
                categories.append(str(label.item()))
                
    return categories

