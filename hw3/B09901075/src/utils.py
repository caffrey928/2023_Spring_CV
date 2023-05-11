import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A_for_vx = np.concatenate((u, np.ones((N, 1)), np.zeros((N, 3)), 
                               -1 * np.multiply(u[:, 0], v[:, 0]).reshape((N, 1)), # -ux * vx
                               -1 * np.multiply(u[:, 1], v[:, 0]).reshape((N, 1)), # -uy * vx
                               -1 * v[:, 0].reshape((N, 1))), axis=1) # -vx
    A_for_vy = np.concatenate((np.zeros((N, 3)), u, np.ones((N, 1)), 
                               -1 * np.multiply(u[:, 0], v[:, 1]).reshape((N, 1)), # -ux * vy
                               -1 * np.multiply(u[:, 1], v[:, 1]).reshape((N, 1)), # -uy * vy
                               -1 * v[:, 1].reshape((N, 1))), axis=1) # -vy
    A = np.concatenate((A_for_vx, A_for_vy), axis=0)

    # # TODO: 2.solve H with A
    _, _, VT = np.linalg.svd(A)
    H = VT.T[:, -1]
    H = H / H[-1]
    H = H.reshape(3, 3)

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(range(xmin, xmax), range(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    N = (xmax - xmin) * (ymax - ymin)
    U = np.concatenate((x.reshape((1, N)), y.reshape((1, N)), np.ones((1, N))), axis=0)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H_inv, U)
        V = np.divide(V, V[-1, :])
        Vx = np.rint(V[0, :].reshape(((ymax-ymin),(xmax-xmin))))
        Vy = np.rint(V[1, :].reshape(((ymax-ymin),(xmax-xmin))))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        x_mask = np.bitwise_and((Vx >= 0), (Vx < w_src))
        y_mask = np.bitwise_and((Vy >= 0), (Vy < h_src))
        mask = np.bitwise_and(x_mask, y_mask)
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # TODO: 6. assign to destination image with proper masking
        valid_Vx = Vx[mask].astype(int)
        valid_Vy = Vy[mask].astype(int)
        dst[y[mask], x[mask]] = src[valid_Vy, valid_Vx]
        # pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H, U)
        V = np.divide(V, V[-1, :])
        Vx = np.rint(V[0, :].reshape(((ymax-ymin),(xmax-xmin))))
        Vy = np.rint(V[1, :].reshape(((ymax-ymin),(xmax-xmin))))
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        x_mask = np.bitwise_and((Vx >= 0), (Vx < w_dst))
        y_mask = np.bitwise_and((Vy >= 0), (Vy < h_dst))
        mask = np.bitwise_and(x_mask, y_mask)
        # TODO: 5.filter the valid coordinates using previous obtained mask
        # TODO: 6. assign to destination image using advanced array indicing
        valid_Vx = Vx[mask].astype(int)
        valid_Vy = Vy[mask].astype(int)
        dst[valid_Vy, valid_Vx, :] = src[mask]
        # pass

    return dst
