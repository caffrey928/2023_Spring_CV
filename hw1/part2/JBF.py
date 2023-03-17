import numpy as np
import cv2
import os

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ## Calculate Spatial Kernel ##
        half_wndw = int((self.wndw_size - 1) / 2)
        x = np.arange(self.wndw_size)
        xx, yy = np.meshgrid(x, x)
        Gs = np.exp((np.square(xx - half_wndw) + np.square(yy - half_wndw)) / (-2 * (self.sigma_s ** 2)))

        ## Apply Spatial Kernel & Range Kernel ##
        padded_img = padded_img.astype(np.float64)
        padded_guidance = padded_guidance.astype(np.float64)
        padded_guidance /= 255

        output = np.empty(img.shape).astype(np.float64)

        w = padded_guidance.shape[0]
        l = padded_guidance.shape[1]
        temp = np.zeros(img.shape).astype(np.float64)
        Gs_Gr_sigma = np.zeros(img.shape).astype(np.float64)

        for i in range(0, self.wndw_size ** 2):
            iter_x = int(i / self.wndw_size)
            iter_y = int(i % self.wndw_size)
            img_x = half_wndw - iter_x
            img_y = half_wndw - iter_y

            if img_x < 0 and img_y < 0:
                distance = padded_guidance[-img_x:w, -img_y: l] - padded_guidance[0:w+img_x, 0:l+img_y]

            elif img_x >= 0 and img_y < 0:
                distance = padded_guidance[0:w-img_x, -img_y:l] - padded_guidance[img_x:w, 0:l+img_y]

            elif img_x < 0 and img_y >= 0:
                distance = padded_guidance[-img_x:w, 0: l-img_y] - padded_guidance[0:w+img_x, img_y:l]

            else:
                distance = padded_guidance[0:w-img_x, 0: l-img_y] - padded_guidance[img_x:w, img_y:l]

            if len(distance.shape) != 3:
                Gr = np.exp(np.square(distance) / (-2 * (self.sigma_r ** 2)))
            else:
                Gr = np.exp(np.square(distance).sum(axis = 2) / (-2 * (self.sigma_r ** 2)))

            Gs_Gr = Gr * Gs[iter_x][iter_y]
            Gs_Gr_expand = np.repeat(Gs_Gr[:, :, np.newaxis], 3, axis=2)

            if img_x < 0 and img_y < 0:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[-img_x:w, -img_y: l])

                temp += Gs_Gr_Ip[half_wndw:w-half_wndw, half_wndw:l-half_wndw]
                Gs_Gr_sigma += Gs_Gr_expand[half_wndw:w-half_wndw, half_wndw:l-half_wndw]

            elif img_x >= 0 and img_y < 0:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[0:w-img_x, -img_y:l])

                temp += Gs_Gr_Ip[iter_x:-half_wndw, half_wndw:l-half_wndw]
                Gs_Gr_sigma += Gs_Gr_expand[iter_x:-half_wndw, half_wndw:l-half_wndw]

            elif img_x < 0 and img_y >= 0:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[-img_x:w, 0: l-img_y])

                temp += Gs_Gr_Ip[half_wndw:w-half_wndw, iter_y:-half_wndw]
                Gs_Gr_sigma += Gs_Gr_expand[half_wndw:w-half_wndw, iter_y:-half_wndw]
            else:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[0:w-img_x, 0:l-img_y])

                temp += Gs_Gr_Ip[iter_x:-half_wndw, iter_y:-half_wndw]
                Gs_Gr_sigma += Gs_Gr_expand[iter_x:-half_wndw, iter_y:-half_wndw]
        
        output = np.divide(temp, Gs_Gr_sigma)

        return np.clip(output, 0, 255).astype(np.uint8)