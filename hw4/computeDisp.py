import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    cost_lr = np.zeros((max_disp + 1, h, w), dtype=np.float32)
    cost_rl = np.zeros((max_disp + 1, h, w), dtype=np.float32)

    padded_Il = cv2.copyMakeBorder(Il, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE)
    padded_Ir = cv2.copyMakeBorder(Ir, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_REPLICATE)
    
    binary_Il = np.zeros((8, h+2, w+2, ch))
    binary_Ir = np.zeros((8, h+2, w+2, ch))
    count = 0

    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            m = (padded_Il > np.roll(padded_Il, [i, j], axis=(0, 1)))
            binary_Il[count][m] = 1
            m = (padded_Ir > np.roll(padded_Ir, [i, j], axis=(0, 1)))
            binary_Ir[count][m] = 1
            count += 1

    binary_Il = binary_Il[:, 1:-1, 1:-1].astype(np.uint8)
    binary_Ir = binary_Ir[:, 1:-1, 1:-1].astype(np.uint8)

    for disp in range(max_disp + 1):
        hamming_cost = np.sum(np.sum(binary_Il[:, :, disp:] ^ binary_Ir[:, :, : w-disp], axis=0), axis=2).astype(np.float32)
        hamming_cost_l = cv2.copyMakeBorder(hamming_cost, top=0, bottom=0, left=disp, right=0, borderType=cv2.BORDER_REPLICATE)
        hamming_cost_r = cv2.copyMakeBorder(hamming_cost, top=0, bottom=0, left=0, right=disp, borderType=cv2.BORDER_REPLICATE)
        cost_lr[disp] = xip.jointBilateralFilter(Il, hamming_cost_l, 22, 10, 26)
        cost_rl[disp] = xip.jointBilateralFilter(Ir, hamming_cost_r, 22, 10, 26)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winner_cost_lr = np.argmin(cost_lr, axis=0)
    winner_cost_rl = np.argmin(cost_rl, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    x, y = np.meshgrid(range(w), range(h))
    x = x - winner_cost_lr
    z = np.zeros(x.shape)
    m = (x >= z)
    check = (winner_cost_lr[m] != winner_cost_rl[y[m], x[m]])
    winner_cost_lr[y[m][check], x[m][check]] = -1

    for x in range(h):
        for y in range(w):
            if winner_cost_lr[x, y] != -1:
                continue
            left = 1
            right = 1
            while y - left >= 0 and winner_cost_lr[x, y-left] == -1:
                left += 1
            if y < left:
                winner_cost_lr[x, y] = max_disp
            else:
                winner_cost_lr[x, y] = winner_cost_lr[x, y-left]
            
            while y + right <= w - 1 and winner_cost_lr[x, y+right] == -1:
                right += 1
            if y + right > w - 1:
                if winner_cost_lr[x, y] > max_disp:
                    winner_cost_lr[x, y] = max_disp
            else:
                if winner_cost_lr[x, y] > winner_cost_lr[x, y+right]:
                    winner_cost_lr[x, y] = winner_cost_lr[x, y+right]
            
    Il = Il.astype(np.uint8)
    winner_cost_lr = winner_cost_lr.astype(np.uint8)
    labels = xip.weightedMedianFilter(Il, winner_cost_lr, 20)

    return labels.astype(np.uint8)