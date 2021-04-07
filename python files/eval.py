import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, target_paths , results_paths):
        self.targets_path = targets_paths
        self.results_path = results_paths
    def make_CM(self, target_image , result_image , thres):
        temp_target_image = np.zeros_like(target_image)
        temp_result_image = np.zeros_like(result_image)
        
        temp_target_image[target_image == 255]  =1

        
        temp_result_image[result_image >= thres] = 1

        
        sm = temp_target_image + temp_result_image
        diff = temp_target_image - temp_result_image
        
        tp = (sm == 2).sum()
        tn = (sm == 0).sum()
        fp = (diff == -1).sum()
        fn = (diff == 1).sum()
        
        return tp,tn,fp,fn
    
    def calc_PR_curve(self, step):
        precision = [0]*step
        recall = [0]*step
        thres_arr = np.linspace(0,1,step)
        for target_path , result_path in zip(self.targets_path,self.results_path):
            target = cv2.imread(target_path,0)
            result = np.load(result_path, allow_pickle=True)
            
        
            for i in range(step):
                thres = thres_arr[i] 
                tp,tn,fp,fn = self.make_CM(target,result,thres)
                if(tp + fp == 0):
                    precision[i] += 1
                else:
                    precision[i] += tp/(tp + fp)
                
                if(tp + fn == 0):
                    recall[i] += 1
                else:
                    recall[i] += tp/(tp + fn)
                    
        precision = [x/len(self.targets_path) for x in precision]
        recall = [x/len(self.targets_path) for x in recall]
        
        return precision , recall, thres_arr
            
    def voc_ap(rec, prec):
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
      if mrec[i] != mrec[i-1]:
        i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

    def plot_loss(loss):
    # Create count of the number of epochs
    epoch_count = range(1, len(loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, loss, 'r--')
    #plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

        
            

    
             
     
