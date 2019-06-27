""" This file tests the accuracy of a segmentation algorithm against 128 labelled examples. It returns the mean IOU metric. 

Google drive link for labelled images:
https://drive.google.com/drive/folders/1YdQIGYjrGnJRZfsoqLq_Nz0OVfOXcqUh?usp=sharing
"""
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import time
import re
# Algorithm python file should be in same directory
from filterkm import get_predicted_labels

# Location of the directory containing images and the .xml label files is stored in IMG_DIR
IMG_DIR = os.getcwd() + '\\export'

def get_mean_IOU(num_files = 40):
    """ Returns mean IOU
    Input:
    num_files- number of image labels to extract 

    Output:
    mean_IOU- mean intersection over union 
    """

    # This dictionary stores the filename and its IOU value, useful for finding files with issues/low IOU.
    fname_IOU_dict = dict()
    
    # Stores all true and predicted labels of the dataset
    All_true_labels, All_predicted_labels, All_image_shapes = list(), list(), list()
    for file in os.listdir(IMG_DIR):
        if file.endswith(".jpeg"):

            # region PART-1: Get the TRUE labels 
            true_label = []
            try:
                with open(os.path.join(IMG_DIR, os.path.splitext(file)[0]+'.xml'), 'r') as f:
                    xmlTree = ET.parse(f)
                    root = xmlTree.getroot()
                    for polygon in root.iter('polygon'):
                        i, temp = 0, []
                        for coordinate in polygon.iter():
                            if ' ' not in coordinate.text:  # To ignore the <polygon> tag
                                temp.append(int(coordinate.text))
                                i += 1
                                if i%2 == 0:
                                    true_label.append(temp)
                                    temp = []
                            # print(coordinate.tag, coordinate.text)

            except Exception as e:
                print(e)
                continue
            #endregion

            # region PART-2: Get the PREDICTED labels using algorithm
            FNAME = os.path.join(IMG_DIR, file)
            predicted_label = get_predicted_labels(FNAME)

            # print(true_label)
            # print(predicted_label)

            # Ordering of points from xml file and cv2.boxPoints needs to match. Changing order of true_label to match the cv2.boxPoints order
            cv2order = [3, 0, 1, 2]
            true_label = [true_label[i] for i in cv2order]

            true_label, predicted_label = np.array(true_label), np.array(predicted_label)
            # print(np.array(true_label).shape, np.array(predicted_label).shape)

            # After changing ordering
            # print(true_label)
            # print(predicted_label)
            #endregion

            # region PART-3: Calculate the IOU
            img = cv2.imread(FNAME)
            iou = get_IOU(true_label, predicted_label, (img.shape[0], img.shape[1]))
            fname_IOU_dict[file] = iou

            All_true_labels.append(true_label)
            All_predicted_labels.append(predicted_label)
            All_image_shapes.append(img.shape)
            # Exit the loop after extracting num_files number of images
            num_files -= 1
            if num_files == 0: break 
            if num_files%20==0: print(f'{num_files} images left to calculate', '\n', '-'*40)


    mean_IOU = np.mean(list(fname_IOU_dict.values()))
    df = pd.DataFrame(list(fname_IOU_dict.items()), columns = ['Filename', 'IOU'])
    df['True Label'] = All_true_labels
    df['Predicted Label'] = All_predicted_labels
    df['Image shape'] = All_image_shapes
    df.to_csv("filename_IOU.csv", index=False)
    #endregion

    return mean_IOU
              
     
def get_IOU(true, pred, img_shape=(1600, 1600)):
    """ 
    Input: numpy arrays of shape (4, 2)
    true- [[x1 y1], [x2 y2], [x3 y3], [x4 y4]]    True labels
    pred- [[x1 y1], [x2 y2], [x3 y3], [x4 y4]]    Prediced labels
    """
    #region openCV solution (It works but it is slower than Shapely)
    # true_mask, pred_mask = np.zeros(img_shape, np.uint8), np.zeros(img_shape, np.uint8)
    # cv2.drawContours(true_mask, [true], -1, 255, -1)
    # cv2.drawContours(pred_mask, [pred], -1, 255, -1)

    # # cv2.imshow('true mask', true_mask)
    # # cv2.imshow('predicted mask', pred_mask)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # intersection = cv2.countNonZero(cv2.bitwise_and(true_mask, pred_mask))
    # union = cv2.countNonZero(cv2.bitwise_or(true_mask, pred_mask))
    # iou = intersection/union
    # print('IOU (opencv):', round(iou, 3))
    # endregion
    
    #region Shapely solution
    from shapely.geometry import Polygon    #conda install shapely
    true_mask = Polygon(true)
    pred_mask = Polygon(pred)
    iou = true_mask.intersection(pred_mask).area / true_mask.union(pred_mask).area
    # print('IOU (shapely):', round(iou, 3))
    print('IOU:', round(iou, 3))
    #endregion
    return iou

def visualize_algo_performance(CSV_fname):
    #** Be careful of the column names in the dataframe. Most errors are because of 
    # not having capitalization in the column names

    """ USER PARAMETERS """
    BAD_THRESHOLD = 0.9     # Below what IOU do we want to visualize the problem
    NUM_IMAGES = 50         # Number of images to visualize of such kind
    IMG_LONGER_DIM = 900    # Length of the longer side of resized image

    df = pd.read_csv(CSV_fname)
    bad_perf_files = df[df.IOU<BAD_THRESHOLD]
    print('Total number of images:', df.shape[0])
    for BAD_THRESHOLD in [0.10, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95]:
        print(f'Number of images below {BAD_THRESHOLD} threshold:', df[df.IOU<BAD_THRESHOLD].shape[0])

    for file in bad_perf_files.Filename[:NUM_IMAGES]:
        FNAME = os.path.join(IMG_DIR, file)
        img = cv2.imread(FNAME)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_data = bad_perf_files.loc[bad_perf_files.Filename == file]

        # Obtain the 2D array of bounding box coordinates from csv file and resize it
        true_box = np.array([int(i) for i in re.findall(r'\d+', img_data['True Label'].values[0])]).reshape((4, 2))
        pred_box = np.array([int(i) for i in re.findall(r'\d+', img_data['Predicted Label'].values[0])]).reshape((4, 2))
        # print(true_box)
        # print(pred_box)

        longer_dim = np.argmax([img.shape[0], img.shape[1]])
        
        if longer_dim == 0:
            x_scale = img.shape[0]/IMG_LONGER_DIM
            y_scale = x_scale
            resize_shape = (IMG_LONGER_DIM, int(IMG_LONGER_DIM*(img.shape[1]/img.shape[0])), 3)
        else:
            x_scale = img.shape[1]/IMG_LONGER_DIM
            y_scale = x_scale
            resize_shape = (int(IMG_LONGER_DIM*(img.shape[0]/img.shape[1])), IMG_LONGER_DIM, 3)

        # print(x_scale, y_scale)
        true_box = true_box/[x_scale, y_scale]
        true_box = true_box.astype(int)
        pred_box = pred_box/[x_scale, y_scale]
        pred_box = pred_box.astype(int)

        # print(true_box)
        # print(pred_box)
        # print('Original Image size:', img.shape)
        # print('Resized Image size :', resize_shape)   

        img = cv2.resize(img, resize_shape[:-1][::-1])
        cv2.drawContours(img, [true_box],0,(0,255,0),2)
        cv2.drawContours(img,[pred_box],0,(0, 0, 255),2)
        cv2.putText(img, 'IOU: ' + str(round(img_data.IOU.values[0], 3)), (resize_shape[1]-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('TRUE(green) and PREDICTED(red) labels',img)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()


def main():
    # region PART-1: Obtain mean IOU
    input('-'*80+'\n\nPlease ensure that the correct path for labelled images folder is given to IMG_DIR constant. Also the algorithm python file is in current working directory.\n\n'+'-'*80)
    
    t = time.time()
    # From how many images should the mean IOU be calculated. 
    num_files = 128
    # If num_files > # of images in directory, it won't matter. For going through all images in directory you can take arbitrary large value for num_files e.g. 1000. We have 128 labelled images. Large images take more time for calculating the IOU. Hence, whenever IOU takes longer, the image is large.

    
    print(f'Mean IOU on {num_files} images: {round(get_mean_IOU(num_files), 3)}')
    t = time.time()-t
    print(f'Time taken to execute {num_files} images:{round(t, 2)} seconds')    
    print(f'Average time taken for single image:{round(t/num_files, 2)} seconds')  
    # endregion

    # region PART-2: Visualize the results
    # Visualize the issues by reading the .csv file
    CSV_fname = 'filename_IOU.csv'
    visualize_algo_performance(CSV_fname)
    # endregion  


if __name__ == '__main__':
    main()
