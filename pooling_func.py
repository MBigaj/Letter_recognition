import numpy as np


def max_pooling(pools: np.array):
    target_shape = (int(np.sqrt(pools.shape[0])), int(np.sqrt(pools.shape[0])))
    max_pools = []
    for pool in pools:
        max_pools.append(np.max(pool))
    return np.array(max_pools).reshape(target_shape)

# def max_pools(pools):
#     num_of_pools = pools.shape[0]
#     target_shape = (int(np.sqrt(num_of_pools)), int(np.sqrt(num_of_pools)))
#     pooled = []
#
#     for pool in pools:
#         pooled.append(np.max(pool))
#     return np.array(pooled).reshape(target_shape)


def return_pools(img, pool_size, stride):
    all_pools = []
    for i in np.arange(img.shape[0], step=stride):
        for j in np.arange(img.shape[0], step=stride):
            single_pool = img[i:i + pool_size, j:j + pool_size]
            if single_pool.shape == (pool_size, pool_size):
                all_pools.append(single_pool)
    return np.array(all_pools)


# def get_pools(img, pool_size, stride):
#     pools = []
#     for i in np.arange(img.shape[0], step=stride):
#         for j in np.arange(img.shape[0], step=stride):
#             mat = img[i:i + pool_size, j:j + pool_size]
#
#             if mat.shape == (pool_size, pool_size):
#                 pools.append(mat)
#     return np.array(pools)