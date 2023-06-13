import cv2
import numpy as np
def cost(thresh):
    num=np.array(np.where(thresh==np.max(thresh))).T
    x_max=np.array(np.where(num==np.max(num[:,1]))).T
    x_min=np.array(np.where(num==np.min(num[:,1]))).T
    y_max=np.array(np.where(num==np.max(num[:,0]))).T
    y_min=np.array(np.where(num==np.min(num[:,0]))).T
    r_p=num[x_max[0,0],:].reshape((1,2))
    l_p=num[x_min[0,0],:].reshape((1,2))
    d_p=num[y_max[0,0],:].reshape((1,2))
    t_p=num[y_min[0,0],:].reshape((1,2))  
    dis_rl= int(np.linalg.norm(r_p-l_p))
    dis_td= int(np.linalg.norm(t_p-d_p))
    v_c=np.zeros((int(dis_td),2))
    h_c=np.zeros((int(dis_rl),2))
    for i in range(int(dis_td)):
        v_c[i,:]=((d_p-t_p)/dis_td)*i+t_p
    for j in range(int(dis_rl)):
        h_c[j,:]=((r_p-l_p)/dis_rl)*j+l_p
    point_predict=np.zeros((int(dis_td)))
    # print(h_c)
    # print(int(dis_td))
    cof=0
    if  (int(dis_td)==0) and (v_c==[]) and (h_c==[]):
        cof=0
    
    elif  int(dis_td)!=0:
        for k in range(int(dis_td)):
            cost_hv=abs(h_c-v_c[k,:].reshape((1,2)).repeat([h_c.shape[0]],axis=0))
            if cost_hv==[]:
                continue
            point_predict[k]=np.min(np.sum(cost_hv,axis=1))
        # if point_predict==[]:
        #     cof=0
        # else:
        # print(point_predict)
        # print(thresh)
        best=np.array(np.where(point_predict==np.min(point_predict)))
        cof=best[0][0]/dis_td
    return cof

def my_awesome_algorithm(image):
    img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_test2=img_gray
    #第一次二值化
    rr=(np.unique(img_gray.reshape(1,np.size(img_gray))))[:20]
    lis_pe=np.array(np.where(img_gray==rr[3])).T
    lis_pe_sum=(np.around(np.mean(lis_pe,axis=0)))
    lis_pe=np.argsort(img_gray)
    img_test2=img_gray
    # if int(lis_pe_sum[1])<=int((img_gray.shape[1])/3):
    #    img_test2[:int(lis_pe_sum[0]),int((img_gray.shape[1])/2):]=0
    #    img_test2[:int(lis_pe_sum[0]),:]=0
    # elif int(lis_pe_sum[1])>=int((img_gray.shape[1])*2/3):
    #     img_test2[:int(lis_pe_sum[0]),:int((img_gray.shape[1])/2)]=0
    #     img_test2[:int(lis_pe_sum[0]),:]=0
    # else:
    #     img_test2[:,:int(lis_pe_sum[1]/3)]=0
    #     img_test2[:,int(img_gray.shape[1]-lis_pe_sum[1]/3):]=0
    #     img_test2[:int(lis_pe_sum[0]),:]=0
    retv,thresh3=cv2.threshold(img_test2,40,255,cv2.THRESH_BINARY_INV)
    retv,thresh0=cv2.threshold(img_gray,35,255,cv2.THRESH_BINARY_INV)
    # thresh3=cv2.medianBlur(thresh3, 5)
    thresh0=cv2.medianBlur(thresh0, 5)
    contours_1, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_ce=500
    mi_y=-1
    for i in range(len(contours_1)):
        (x, y), r = cv2.minEnclosingCircle(contours_1[i])
        if (int(y)+int(r/4)>=img_gray.shape[0]) or (int(x)+int(r/4)>=img_gray.shape[1]):
            continue
        if (img_gray[int(y),int(x)]<min_ce) :
            min_ce=img_gray[int(y),int(x)]
            mi_x=int(x)
            mi_y=int(y)
            mi_r=int(r)
    #canny邊緣偵測
    gg=cv2.Canny(thresh0, 35,80)
    gg = cv2.GaussianBlur(gg, (15,15), 100)
    #霍夫圓偵測
    cir=cv2.HoughCircles(gg,cv2.HOUGH_GRADIENT_ALT,0.5,20,param1=50,param2=0.5,minRadius=20,maxRadius=120)

    if ( cir is not None ) and (mi_y!=-1):
        circles = np.uint16(np.around(cir))
        #篩選最佳圓
        min_val=500
        # min_cir=-1
        for i in circles[0,:]:
            if img_gray[i[1],i[0]]<min_val:
                min_val=img_gray[i[1],i[0]]
                min_cir=i
        #框出圓位置
        mask=np.zeros(img_gray.shape)
        y_m=min_cir[1]
        x_m=min_cir[0]
        d_m=min_cir[2]+15
        mask[y_m-d_m:y_m+d_m,x_m-d_m:x_m+d_m]=1
        pp=mask*img_gray
        # cv2_imshow(pp)
        #第二次二值化 算cost
        retv,thresh4=cv2.threshold(pp,40,255,cv2.THRESH_BINARY_INV)
        thresh4[:y_m-d_m,:]=0
        thresh4[y_m-d_m:,:x_m-d_m]=0
        thresh4[y_m-d_m:,x_m+d_m:]=0
        thresh4[y_m+d_m:,x_m-d_m:x_m+d_m]=0
        # cv2_imshow(thresh4)
        # if np.max(thresh4)==0:
        #     conf=0
        # else:
        #     conff=cost(thresh4)
        #     conf=conff*2
        #     if conff>=0.4:
        #         conf=1
        #描出局部邊緣
        contours_2,hierarchy=cv2.findContours(np.array(thresh4,np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #橢圓擬和 篩選最佳橢圓
        # min_ell=500
        if (len(contours_2)==0) and (mi_y!=-1):
            #框出圓位置，霍夫中方法找不到邊緣，用法一有值
            mask=np.zeros(img_gray.shape)
            y_m=mi_y
            x_m=mi_x
            d_m=mi_r
            mask[y_m-d_m:y_m+d_m,x_m-d_m:x_m+d_m]=1
            pp=mask*img_gray
            #第二次二值化 算cost
            retv,thresh4=cv2.threshold(pp,40,255,cv2.THRESH_BINARY_INV)
            thresh4[:y_m-d_m,:]=0
            thresh4[y_m-d_m:,:x_m-d_m]=0
            thresh4[y_m-d_m:,x_m+d_m:]=0
            thresh4[y_m+d_m:,x_m-d_m:x_m+d_m]=0
            # cv2_imshow(thresh4)
            if  (d_m<15):
                conf=0
            else:
                conff=cost(thresh4)
                conf=conff*2
                if conff>=0.4:
                    conf=1
            contours_3,hierarchy=cv2.findContours(np.array(thresh4,np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            final=-1
            min_ell=500
            for i in range(len(contours_3)):
                cnt = np.array(contours_3[i])
                if cnt.shape[0]<30:
                    continue
                ellipse = cv2.fitEllipse(cnt)
                if (int((ellipse[0][1]))>=img_gray.shape[0]) or (int((ellipse[0][0]))>=img_gray.shape[1]) or (int((ellipse[0][1]))<0) or (int((ellipse[0][0]))<0):
                    continue
                if img_gray[int((ellipse[0][1])),int((ellipse[0][0]))] < min_ell:
                    min_ell=img_gray[int((ellipse[0][1])),int((ellipse[0][0]))]
                    final=i
            if final==-1:
               mask_final=np.zeros((img_gray.shape))
               conf=0
            else:
                con=contours_3[final]
                bl=np.zeros((img_gray.shape))
                ellipse = cv2.fitEllipse(con)
                cv2.ellipse(bl, ellipse, (255, 255, 0), 1)
                mask=np.array(np.where(bl==np.max(bl))).T
                #橢圓內填值
                mask_final=bl
                lis = np.unique(mask[:,0]) 
                for i in range(len(lis)):
                    lis_p=np.array(np.where(mask[:,0]==lis[i]))
                    mask_final[mask[lis_p[0][0],0],mask[lis_p[0][0],1]:mask[lis_p[0][len(lis_p[0])-1],1]]=np.max(mask_final) 
        elif len(contours_2)!=0 and (mi_y==-1):
                    # print('cc',contours)
            min_val=500
            for i in circles[0,:]:
                if img_gray[i[1],i[0]]<min_val:
                    min_val=img_gray[i[1],i[0]]
                    min_cir=i
            #框出圓位置
            mask=np.zeros(img_gray.shape)
            y_m=min_cir[1]
            x_m=min_cir[0]
            d_m=min_cir[2]+15
            mask[y_m-d_m:y_m+d_m,x_m-d_m:x_m+d_m]=1
            pp=mask*img_gray
            # cv2_imshow(pp)
            #第二次二值化 算cost
            retv,thresh4=cv2.threshold(pp,40,255,cv2.THRESH_BINARY_INV)
            thresh4[:y_m-d_m,:]=0
            thresh4[y_m-d_m:,:x_m-d_m]=0
            thresh4[y_m-d_m:,x_m+d_m:]=0
            thresh4[y_m+d_m:,x_m-d_m:x_m+d_m]=0
            if  (d_m<15):
                conf=0
            else:
                conff=cost(thresh4)
                conf=conff*2
                if conff>=0.4:
                    conf=1
            min_ell=500
            final=-1
            for i in range(len(contours_2)):
                cnt = np.array(contours_2[i])
                if cnt.shape[0]<15:
                    continue
                ellipse = cv2.fitEllipse(cnt)
                if (int((ellipse[0][1]))>=img_gray.shape[0]) or (int((ellipse[0][0]))>=img_gray.shape[1]) or (int((ellipse[0][1]))<0) or (int((ellipse[0][0]))<0):
                    continue
                if img_gray[int((ellipse[0][1])),int((ellipse[0][0]))] < min_ell:
                    min_ell=img_gray[int((ellipse[0][1])),int((ellipse[0][0]))]
                    final=i
            if final==-1:
                mask_final=np.zeros((img_gray.shape))
                conf=0
            else:
                con=contours_2[final]
                bl=np.zeros((img_gray.shape))
                ellipse = cv2.fitEllipse(con)
                cv2.ellipse(bl, ellipse, (255, 255, 0), 1)

                mask=np.array(np.where(bl==np.max(bl))).T

                #橢圓內填值
                mask_final=bl
                lis = np.unique(mask[:,0]) 
                for i in range(len(lis)):
                    lis_p=np.array(np.where(mask[:,0]==lis[i]))
                    mask_final[mask[lis_p[0][0],0],mask[lis_p[0][0],1]:mask[lis_p[0][len(lis_p[0])-1],1]]=np.max(mask_final) 
        elif len(contours_2)!=0 and (mi_y!=-1):
            #框出圓位置，霍夫中方法找不到邊緣，用法一有值
            mask=np.zeros(img_gray.shape)
            y_m=mi_y
            x_m=mi_x
            d_m=mi_r
            mask[y_m-d_m:y_m+d_m,x_m-d_m:x_m+d_m]=1
            pp=mask*img_gray
            #第二次二值化 算cost
            retv,thresh4=cv2.threshold(pp,40,255,cv2.THRESH_BINARY_INV)
            thresh4[:y_m-d_m,:]=0
            thresh4[y_m-d_m:,:x_m-d_m]=0
            thresh4[y_m-d_m:,x_m+d_m:]=0
            thresh4[y_m+d_m:,x_m-d_m:x_m+d_m]=0
            # cv2_imshow(thresh4)
            conf=1

                    
            contours_5,hierarchy=cv2.findContours(np.array(thresh4,np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            final=-1
            min_ell=500
            for i in range(len(contours_5)):
                cnt = np.array(contours_5[i])
                if cnt.shape[0]<30:
                    continue
                ellipse = cv2.fitEllipse(cnt)
                if (int((ellipse[0][1]))>=img_gray.shape[0]) or (int((ellipse[0][0]))>=img_gray.shape[1]) or (int((ellipse[0][1]))<0) or (int((ellipse[0][0]))<0):
                    continue
                if img_gray[int((ellipse[0][1])),int((ellipse[0][0]))] < min_ell:
                    min_ell=img_gray[int((ellipse[0][1])),int((ellipse[0][0]))]
                    final=i
            if final==-1:
               mask_final=np.zeros((img_gray.shape))
               conf=0
            else:
                con=contours_5[final]
                bl=np.zeros((img_gray.shape))
                ellipse = cv2.fitEllipse(con)
                cv2.ellipse(bl, ellipse, (255, 255, 0), 1)
                mask=np.array(np.where(bl==np.max(bl))).T
                #橢圓內填值
                mask_final=bl
                lis = np.unique(mask[:,0]) 
                for i in range(len(lis)):
                    lis_p=np.array(np.where(mask[:,0]==lis[i]))
                    mask_final[mask[lis_p[0][0],0],mask[lis_p[0][0],1]:mask[lis_p[0][len(lis_p[0])-1],1]]=np.max(mask_final)


        else:
            conf=0
            mask_final=np.zeros((img_gray.shape))
    elif (cir is None) and (mi_y!=-1):
        #框出圓位置，霍夫中方法找不到圓cir無，法一有值
        mask=np.zeros(img_gray.shape)
        y_m=mi_y
        x_m=mi_x
        d_m=mi_r
        mask[y_m-d_m:y_m+d_m,x_m-d_m:x_m+d_m]=1
        pp=mask*img_gray
        #第二次二值化 算cost
        retv,thresh4=cv2.threshold(pp,40,255,cv2.THRESH_BINARY_INV)
        thresh4[:y_m-d_m,:]=0
        thresh4[y_m-d_m:,:x_m-d_m]=0
        thresh4[y_m-d_m:,x_m+d_m:]=0
        thresh4[y_m+d_m:,x_m-d_m:x_m+d_m]=0
        # print(mi_y,mi_x,mi_r)
        if  d_m<=15:
            conf=0
        elif d_m>15:
            conf=1
        # else:
        #     conff=cost(thresh4)
        #     conf=conff*2
        #     if conff>=0.4:
        #         conf=1
        contours_4,hierarchy=cv2.findContours(np.array(thresh4,np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        final=-1
        min_ell=500
        for i in range(len(contours_4)):
            cnt = np.array(contours_4[i])
            if cnt.shape[0]<30:
                continue
            ellipse = cv2.fitEllipse(cnt)
            if (int((ellipse[0][1]))>=img_gray.shape[0]) or (int((ellipse[0][0]))>=img_gray.shape[1]) or (int((ellipse[0][1]))<0) or (int((ellipse[0][0]))<0):
                continue
            if img_gray[int((ellipse[0][1])),int((ellipse[0][0]))] < min_ell:
                min_ell=img_gray[int((ellipse[0][1])),int((ellipse[0][0]))]
                final=i
        if final==-1:
            mask_final=np.zeros((img_gray.shape))
            conf=0
        else:
            con=contours_4[final]
            bl=np.zeros((img_gray.shape))
            ellipse = cv2.fitEllipse(con)
            cv2.ellipse(bl, ellipse, (255, 255, 0), 1)
            mask=np.array(np.where(bl==np.max(bl))).T
            #橢圓內填值
            mask_final=bl
            lis = np.unique(mask[:,0]) 
            for i in range(len(lis)):
                lis_p=np.array(np.where(mask[:,0]==lis[i]))
                mask_final[mask[lis_p[0][0],0],mask[lis_p[0][0],1]:mask[lis_p[0][len(lis_p[0])-1],1]]=np.max(mask_final) 
    else:
        conf=0
        mask_final=np.zeros((img_gray.shape))
    # if np.max(mask_final)==0:
    #     conf=0
    # else:
    #     conf=1
    return mask_final,conf