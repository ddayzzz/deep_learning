import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # 构成 full 矩阵, -1 是因为原图的左上角的点占据了一个位置
    temp_mat = np.zeros((Hi + Hk - 1, Wi + Wk - 1))
    for ii in range(Hi + Hk - 1):
        for ij in range(Wi + Wk - 1):
            result = 0
            for ki in range(Hk):
                for kj in range(Wk):
                    # 计算以 kernel 为中心的原图的卷积后的值, 可以参考 slide
                    if ii - ki >= 0 and ii - ki < Hi and ij - kj >= 0 and ij - kj < Wi:
                        result += kernel[ki][kj] * image[ii - ki][ij - kj]
            temp_mat[ii][ij] = result
    # 裁剪 same 区域
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = temp_mat[int(i + (Hk - 1) / 2)][int(j + (Wk - 1) / 2)]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    zeros_w = np.zeros([H, pad_width])
    out = np.hstack([zeros_w, image, zeros_w])
    zeros_h = np.zeros([pad_height, W + 2 * pad_width])
    out = np.vstack([zeros_h, out, zeros_h])
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_height = Hk // 2
    pad_width = Wk // 2
    image_padding = zero_pad(image, pad_height, pad_width)
    # 卷积之前需要分别按照 X, Y 方向翻转
    kernel_flip = np.flip(np.flip(kernel, 0), 1)

    for i in range(Hi):
        for j in range(Wi):
            # 直接求邻近元素的点积
            out[i][j] = np.sum(np.multiply(kernel_flip, image_padding[i:(i + Hk), j:(j + Wk)]))

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = (g - np.mean(g)) / np.std(g)
    f = (f - np.mean(f)) / np.std(f)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out
