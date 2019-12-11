from os.path import normpath as fn  # Fixes window/linux path conventions

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from skimage.color import rgb2yuv, yuv2rgb
from skimage.io import imread, imsave

eps = np.finfo(np.float64).eps


def build_weights_matrix(Y, colored, k):
    h, w = Y.shape
    s = h * w
    padding = k // 2
    coord = np.array(np.meshgrid(range(h), range(w))).T.reshape(h, w, 2)
    W = sparse.lil_matrix((s, s))
    for r in range(h):
        for c in range(w):
            center_idx = r * w + c
            if center_idx in colored:
                W[center_idx] = sparse.csr_matrix(([1.0], ([0], [center_idx])), shape=(1, s))
            else:
                # compute neighbour coordinate
                min_y, max_y = max(r - padding, 0), min(r + padding + 1, h)
                min_x, max_x = max(c - padding, 0), min(c + padding + 1, w)
                co = coord[min_y: max_y, min_x: max_x].reshape(-1, 2)
                p_idx = co[:, 0] * w + co[:, 1]
                mask = p_idx != center_idx  # neighbour index

                # compute weight
                n = Y[min_y: max_y, min_x:max_x].flatten()[mask]
                var_r = np.var(n)
                # mean_r = np.mean(n)
                # weights = (1 + 1 / (var_r + eps) * (Y[r, c] - mean_r) * (n - mean_r)).flatten()
                weights = np.exp(-np.square(Y[r, c] - n) / (2 * var_r + eps))
                norm = np.linalg.norm(weights, ord=1)
                # assign to sparse matrix
                W[center_idx, p_idx[mask]] = -1 * np.divide(weights, norm, out=np.zeros_like(weights), where=norm != 0)
    # finally, assign the weight for all pixel center as 1
    W[np.arange(s), np.arange(s)] = 1
    return W.tocsc()


def colorization(img, marked_img, k):
    H, W, C = img.shape

    # convert to YUV color space
    img_yuv = rgb2yuv(img)
    marked_yuv = rgb2yuv(marked_img)

    Y = np.array(img_yuv[:, :, 0])

    # find the color position and update sparse matrix
    diff = np.where(marked_yuv != img_yuv)
    colored_coord_idx = np.unique(diff[0] * W + diff[1])

    weight_matrix = build_weights_matrix(Y, colored_coord_idx, k)

    # compute the least-square solution, by computing the pseudo-inverse
    LU = splu(weight_matrix)

    b1 = (marked_yuv[:, :, 1]).flatten()
    b2 = (marked_yuv[:, :, 2]).flatten()

    U = LU.solve(b1)
    V = LU.solve(b2)

    sol = np.zeros_like(img)
    sol[:, :, 0] = Y
    sol[:, :, 1] = U.reshape((H, W))
    sol[:, :, 2] = V.reshape((H, W))

    return sol


# img_path = fn("samples/flag.bmp")
# mark_img_path = fn("samples/flag_marked.bmp")
# out_img_path = "output/flag_out_"

img_path = fn("samples/baby.bmp")
mark_img_path = fn("samples/baby_marked.bmp")
out_img_path = "output/baby_out_"

# img_path = fn("samples/smiley.bmp")
# mark_img_path = fn("samples/smiley_marked.bmp")
# out_img_path = "output/smiley_out_"

img = np.float32(imread(img_path))
marked_img = np.float32(imread(mark_img_path))

out = colorization(img, marked_img, 5)

imsave(fn(out_img_path + "kernel_5.png"), yuv2rgb(out))
