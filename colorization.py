from os.path import normpath as fn  # Fixes window/linux path conventions

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from skimage.color import rgb2yuv, yuv2rgb
from skimage.io import imread, imsave
from sklearn.preprocessing import normalize

eps = np.finfo(np.float64).eps


def build_weights_matrix(Y, k):
    h, w = Y.shape
    s = h * w
    padding = k // 2
    coord = np.array(np.meshgrid(range(h), range(w))).T.reshape(h, w, 2)
    W = sparse.lil_matrix((s, s))
    for r in range(h):
        for c in range(w):
            # compute neighbour coordinate
            min_y, max_y = max(r - padding, 0), min(r + padding + 1, h)
            min_x, max_x = max(c - padding, 0), min(c + padding + 1, w)
            co = coord[min_y: max_y, min_x: max_x].reshape(-1, 2)
            p_idx = co[:, 0] * w + co[:, 1]

            # compute weight
            n = Y[min_y: max_y, min_x:max_x]
            var_r = np.var(n)
            # mean_r = np.mean(n)
            # weights = (1 + 1 / (var_r + eps) * (Y[r, c] - mean_r) * (n - mean_r)).flatten()
            weights = np.exp(-np.square(Y[r, c] - n) / (2 * var_r + eps)).flatten()
            # assign to sparse matrix
            W[r * w + c, p_idx] = -1 * weights
    # assign all the center to be 0 first, then normalize the weight for the neighbours of each centers
    # finally, assign the weight for all pixel center as 1
    W[np.arange(s), np.arange(s)] = 0
    Wn = normalize(W, norm='l1', axis=1).tolil()
    Wn[np.arange(s), np.arange(s)] = 1
    return Wn


# img_path = fn("samples/flag.bmp")
# mark_img_path = fn("samples/flag_marked.bmp")
# out_img_path = "output/flag_out_"

img_path = fn("samples/baby.bmp")
mark_img_path = fn("samples/baby_marked.bmp")
out_img_path = "output/baby_out_"

img = np.float32(imread(img_path))
marked_img = np.float32(imread(mark_img_path))

H, W, C = img.shape

# convert to YUV color space
img_yuv = rgb2yuv(img)
marked_yuv = rgb2yuv(marked_img)

Y = np.array(img_yuv[:, :, 0])

# find the color position and update sparse matrix
diff = np.where(marked_yuv != img_yuv)
colored_coord_idx = np.unique(diff[0] * W + diff[1])

kernel = [3, 5, 7]

for k in kernel:
    Wn = build_weights_matrix(Y, k)

    for pos in colored_coord_idx:
        Wn[pos] = sparse.csr_matrix(([1.0], ([0], [pos])), shape=(1, H * W))

    # compute the least-square solution, by computing the pseudo-inverse
    LU = splu(Wn.tocsc())

    b1 = (marked_yuv[:, :, 1]).flatten()
    b2 = (marked_yuv[:, :, 2]).flatten()

    U = LU.solve(b1)
    V = LU.solve(b2)

    sol = np.zeros(np.shape(img))
    sol[:, :, 0] = Y
    sol[:, :, 1] = U.reshape((H, W))
    sol[:, :, 2] = V.reshape((H, W))

    imsave(fn(out_img_path + "kernel_{}.png".format(k)), yuv2rgb(sol))