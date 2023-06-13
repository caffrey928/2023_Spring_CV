import cv2 
import numpy as np
from skimage.feature import canny

'''
Requirement: numpy, cv2==4.7.0.72, scikit-image==0.21.0
'''

def refine_pupil(img: np.ndarray, pred: np.ndarray) -> np.ndarray:
    '''
    Argument:  
    img: 3D ndarray
    pred: 2D ndarray 
    
    Return: 
    final_refine: 3D ndarray
    '''
    
    if np.sum(pred) ==0:
        pred = np.expand_dims(pred, axis=2)
        pred = np.dstack((np.zeros_like(pred),pred, np.zeros_like(pred)))
        return pred
    ksize = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    cared_mask = cv2.morphologyEx(pred, cv2.MORPH_DILATE, ksize, iterations=3)>0
    med = cv2.cvtColor(cv2.medianBlur(img, 15), cv2.COLOR_BGR2GRAY)
    big_med = cv2.cvtColor(cv2.medianBlur(img, 27), cv2.COLOR_BGR2GRAY)
    
    his_mean = med[np.bitwise_and(cared_mask, med<med[cared_mask].mean()+5)].mean()
    noise_mask =  np.bitwise_and(cared_mask, med>(his_mean+50))
    final_med = np.where(noise_mask, big_med.astype(np.uint8), med).astype(np.uint8)

    final_his = cv2.equalizeHist(final_med)
    final_edges = canny(final_his, 2.0, 10, 12)
    edges = np.where(cared_mask, final_edges, 0).astype(np.uint8)
    edges_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ksize,  iterations=8)  # ! iteration at least : 8
    contour, hierarchy = cv2.findContours(edges_close, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hie = np.array(hierarchy[0])
        
    cnt = []
    for c , hie in zip(contour,hie):
        if len(cnt) < len(c) and hie[-1] == -1:
            cnt = c 
    if len(cnt) < 5:
        pred = np.expand_dims(pred, axis=2)
        pred = np.dstack((np.zeros_like(pred),pred, np.zeros_like(pred)))
        return pred
    ellipse = cv2.fitEllipse(cnt)
    refine = np.zeros_like(img)
    final_refine = cv2.ellipse(refine, ellipse ,[0,255,0], -1)
    
    return  final_refine
