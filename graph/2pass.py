import numpy as np

def two_pass(im):  # labeling algorithm
    """
    :param im: binary input image
    :return: labeling result
    """

    assert 2 == len(im.shape)
    M, N = im.shape

    label = np.zeros((M, N))
    l_cnt = 0

    merge_list = {}
    for r in xrange(M):
        for c in xrange(N):
            if im[r, c] != 0:
                left, right, up = (0, 0, 0)
                if r and im[r-1, c]: up = label[r-1, c]
                if c and im[r, c-1]: left = label[r, c-1]
                if c < N-1 and im[r, c+1]: right = label[r, c+1]
                if left or right or up:
                    min_label = l_cnt+1 # find the minimum label
                    if left and min_label > merge_list[left]: min_label = merge_list[left]
                    if right and min_label > merge_list[right]: min_label = merge_list[right]
                    if up and min_label > merge_list[up]: min_label = merge_list[up]
                    assert min_label != l_cnt+1
                    label[r, c] = min_label

                    if left:
                        v = merge_list[left]
                        for key, value in merge_list.iteritems():
                            if value == v: merge_list[key] = min_label
                    if right:
                        v = merge_list[right]
                        for key, value in merge_list.iteritems():
                            if value == v: merge_list[key] = min_label
                    if up:
                        v = merge_list[up]
                        for key, value in merge_list.iteritems():
                            if value == v: merge_list[key] = min_label
                else:
                    l_cnt += 1
                    label[r, c] = l_cnt
                    merge_list[l_cnt] = l_cnt # in merge_list, key >= value

    for r in xrange(M):
        for c in xrange(N):
            if im[r, c]:
                min_label = merge_list[label[r, c]]
                label[r, c] = min_label

    return label


def test_two_pass():
    # naive test
    im = np.array([[0,0,1,0,0],
                   [0,1,0,1,0],
                   [0,0,0,1,1],
                   [0,1,1,1,0],
                   [0,0,1,0,1]])
    label = two_pass(im)
    print label

if __name__ == "__main__":
    test_two_pass()
