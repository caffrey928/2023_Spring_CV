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

        ### TODO ###
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
        # for i in range(output.shape[0]):
        #     x = i + half_wndw
        #     for j in range(output.shape[1]):
        #         y = j + half_wndw
                
        #         ## Calculate Gr ##
        #         Tp = padded_guidance[x, y]
        #         Tq = padded_guidance[x - half_wndw : x + half_wndw + 1, y - half_wndw : y + half_wndw + 1]
        #         distance = Tp - Tq
                
        #         if distance.shape != (self.wndw_size, self.wndw_size, 3):
        #             Gr = np.exp(np.square(distance) / (-2 * (self.sigma_r ** 2)))
        #         else:
        #             Gr = np.exp(np.sum(np.square(distance), axis = 2) / (-2 * (self.sigma_r ** 2)))
                
        #         ## Apply bilateral filter ##
        #         Ip = padded_img[x - half_wndw : x + half_wndw + 1, y - half_wndw : y + half_wndw + 1]
        #         Gs_Gr = np.multiply(Gs, Gr)
        #         sigma_Gs_Gr = np.multiply(Gs, Gr).sum(axis=1).sum(axis=0)
                
        #         for k in range(output.shape[2]):
        #             output[i, j, k] = np.multiply(Gs_Gr, Ip[:,:,k]).sum(axis=1).sum(axis=0) / sigma_Gs_Gr

        padded_img_w = padded_guidance.shape[0]
        padded_img_l = padded_guidance.shape[1]
        temp = np.zeros(padded_img.shape).astype(np.float64)
        Gs_Gr_sigma = np.zeros(padded_img.shape).astype(np.float64)

        for i in range(0, self.wndw_size ** 2):
            iter_x = int(i / self.wndw_size)
            iter_y = int(i % self.wndw_size)
            img_x = half_wndw - iter_x
            img_y = half_wndw - iter_y

            if img_x < 0 and img_y < 0:
                distance = padded_guidance[-img_x:padded_img_w, -img_y: padded_img_l] - padded_guidance[0:padded_img_w+img_x, 0:padded_img_l+img_y]
            elif img_x >= 0 and img_y < 0:
                distance = padded_guidance[0:padded_img_w-img_x, -img_y: padded_img_l] - padded_guidance[img_x:padded_img_w, 0:padded_img_l+img_y]
            elif img_x < 0 and img_y >= 0:
                distance = padded_guidance[-img_x:padded_img_w, 0: padded_img_l-img_y] - padded_guidance[0:padded_img_w+img_x, img_y:padded_img_l]
            else:
                distance = padded_guidance[0:padded_img_w-img_x, 0: padded_img_l-img_y] - padded_guidance[img_x:padded_img_w, img_y:padded_img_l]

            if len(distance.shape) != 3:
                Gr = np.exp(np.square(distance) / (-2 * (self.sigma_r ** 2)))
            else:
                Gr = np.exp(np.sum(np.square(distance), axis = 2) / (-2 * (self.sigma_r ** 2)))

            Gs_Gr = Gr * Gs[iter_x][iter_y]
            Gs_Gr_expand = np.repeat(Gs_Gr[:, :, np.newaxis], 3, axis=2)

            if img_x < 0 and img_y < 0:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[-img_x:padded_img_w, -img_y: padded_img_l])

                temp[0:padded_img_w+img_x, 0:padded_img_l+img_y] += Gs_Gr_Ip
                Gs_Gr_sigma[0:padded_img_w+img_x, 0:padded_img_l+img_y] += Gs_Gr_expand

            elif img_x >= 0 and img_y < 0:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[0:padded_img_w-img_x, -img_y: padded_img_l])

                temp[img_x:padded_img_w, 0:padded_img_l+img_y] += Gs_Gr_Ip
                Gs_Gr_sigma[img_x:padded_img_w, 0:padded_img_l+img_y] += Gs_Gr_expand

            elif img_x < 0 and img_y >= 0:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[-img_x:padded_img_w, 0: padded_img_l-img_y])

                temp[0:padded_img_w+img_x, img_y:padded_img_l] += Gs_Gr_Ip
                Gs_Gr_sigma[0:padded_img_w+img_x, img_y:padded_img_l] += Gs_Gr_expand
            else:
                Gs_Gr_Ip = np.multiply(Gs_Gr_expand, padded_img[0:padded_img_w-img_x, 0: padded_img_l-img_y])

                temp[img_x:padded_img_w, img_y:padded_img_l] += Gs_Gr_Ip
                Gs_Gr_sigma[img_x:padded_img_w, img_y:padded_img_l] += Gs_Gr_expand
        
        output = np.divide(temp, Gs_Gr_sigma)[half_wndw:padded_img_l-half_wndw, half_wndw:padded_img_w-half_wndw]

        return np.clip(output, 0, 255).astype(np.uint8)