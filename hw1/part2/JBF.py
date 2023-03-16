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
        Gs = np.exp(np.divide(np.square(xx - half_wndw) + np.square(yy - half_wndw), -2 * (self.sigma_s ** 2)))
        ## Apply Spatial Kernel & Range Kernel ##
        padded_img = padded_img.astype(np.float64)
        padded_guidance = padded_guidance.astype(np.float64)
        padded_guidance /= 255


        output = np.empty(img.shape).astype(np.float64)
        for i in range(output.shape[0]):
            x = i + self.pad_w
            for j in range(output.shape[1]):
                print(padded_guidance.shape[1]-self.pad_w)
                print(output.shape[0] + self.pad_w)
                os._exit()
                y = j + self.pad_w

                Tp = padded_guidance[x][y]
                Tq = padded_guidance[x - half_wndw : (x + half_wndw + 1), y - half_wndw : (y + half_wndw + 1)]
                
                if (Tp - Tq).shape != (self.wndw_size, self.wndw_size, 3):
                    Gr = np.exp(np.divide(np.square(Tp - Tq), -2 * (self.sigma_r ** 2)))
                else:
                    Gr = np.exp(np.divide(np.sum(np.square(Tp - Tq), axis = 2), -2 * (self.sigma_r ** 2)))

                Ip = padded_img[x - half_wndw : x + half_wndw + 1, x - half_wndw : x + half_wndw + 1]
                # Iq = padded_img[i-half_wndw:i+half_wndw+1, j-half_wndw:j+half_wndw+1]
                # print(x)
                # print(i)
                # print(Ip.shape)
                # print(Ip[:][:][0].shape)
                # os._exit()
                kernel = np.multiply(Gs, Gr)
                kk = np.expand_dims(kernel, axis=-1)
                ex_kernel = np.concatenate((np.concatenate((kk, kk), axis=-1), kk), axis=-1)
                output[i][j] = np.divide(np.sum(np.sum(np.multiply(ex_kernel, Ip), axis=0), axis=0), np.sum(kernel))
        # for i in range(self.pad_w, padded_guidance.shape[0]-self.pad_w):
        #     for j in range(self.pad_w, padded_guidance.shape[1]-self.pad_w):
        #         #Step 3-1: Calculus Gr with guidance(gray)
        #         Tp = padded_guidance[i,j]
        #         Tq = padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
        #         pw = np.divide(np.square(Tp-Tq), -2*self.sigma_r*self.sigma_r)
        #         if len(pw.shape)==3:
        #             pw = pw.sum(axis=2)  
        #         Gr=np.exp(pw)

        #         #Step 3-2: Calculus G:=Gr*Gs
        #         G =np.multiply(Gs, Gr)
        #         W =G.sum(axis=1).sum(axis=0)

        #         #Step 3-3: JB-filter I->I'  
        #         Iq=padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]        
        #         for k in range(img.shape[2]):
        #             output[i-self.pad_w, j-self.pad_w, k] = np.multiply(G,Iq[:,:,k]).sum(axis=1).sum(axis=0)/W

        output.astype(np.uint8)
        return np.clip(output, 0, 255).astype(np.uint8)