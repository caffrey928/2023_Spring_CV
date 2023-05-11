import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])
    w = 0

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        num_of_matches = 50
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)[:num_of_matches]
        
        v = np.array([ kp1[match.queryIdx].pt for match in matches ]).reshape(-1, 2)
        u = np.array([ kp2[match.trainIdx].pt for match in matches ]).reshape(-1, 2)
        # TODO: 2. apply RANSAC to choose best H
        iteration = 10000
        num_points_for_H = 4
        max_num_of_inliers = 0
        thres = 0.5
        best_H = np.eye(3)

        for _ in range(iteration):
            index = random.sample(range(num_of_matches), num_points_for_H)
            rand_u, rand_v = u[index], v[index]
            H = solve_homography(rand_u, rand_v)

            U = np.concatenate((u.T, np.ones((1,num_of_matches))), axis=0)
            pred_V = np.dot(H, U)
            pred_V = np.divide(pred_V, pred_V[-1, :])[:-1, :]

            error = np.linalg.norm((v.T - pred_V), axis=0)
            num_of_inliers = sum(error < thres)

            if max_num_of_inliers < num_of_inliers:
                max_num_of_inliers = num_of_inliers
                best_H = H

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)
        # TODO: 4. apply warping
        w += im1.shape[1]
        out = warping(im2, dst, last_best_H, 0, h_max, w, w + im2.shape[1], direction='b')
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)