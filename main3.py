import numpy as np
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
from skimage import draw

"""
image  = (N, N), value is distance.
centers = tuple of centers.

"""

# configure ----------------------------------------------------------------------

# maximum_val
N_max = np.inf


def euclidean_distance(*xys):
    """
    :param xys: (x1, x2, y1, y2)
    :return:
    """
    distance = np.sqrt((xys[0] - xys[1]) ** 2 + (xys[2] - xys[3]) ** 2)
    return distance


def cityblock_distance(*xys):
    """
    :param xys: (x1, x2, y1, y2)
    :return:
    """
    distance = np.abs(xys[0] - xys[1]) + np.abs(xys[2] - xys[3])
    return distance


def chessboard_distance(*xys):
    """
    :param xys: (x1, x2, y1, y2)
    :return:
    """
    distance = np.max([np.abs(xys[0] - xys[1]), np.abs(xys[2] - xys[3])])
    return distance


def quasi_euclidean_distance(*xys):
    """
    :param xys: (x1, x2, y1, y2)
    :return:
    """
    if np.abs(xys[2] - xys[3]) > np.abs(xys[0] - xys[1]):
        distance = np.abs(xys[2] - xys[3]) + (np.sqrt(2) - 1) * np.abs(xys[0] - xys[1])
    else:
        distance = (np.sqrt(2) - 1) * np.abs(xys[2] - xys[3]) + np.abs(xys[0] - xys[1])
    return distance


# distance type.
distance_type = {"D_E": euclidean_distance,
                 "D_4": cityblock_distance,
                 "D_8": chessboard_distance,
                 "D_QE": quasi_euclidean_distance,
                 }


# method ----------------------------------------------------------------------

def generate_image(use_sample=True, init_img=None):
    # TODO: generate random image.

    sample = np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 1, 1, 0, 0, 0, 1, 0],
                       [0, 1, 0, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 0]
                       ], dtype=np.float)
    #
    # sample = np.zeros([80, 80])
    # sample[39, 39] = 1

    if use_sample:
        data = sample
    else:
        data = np.zeros([init_img["size"], init_img["size"]])

        x_coord = np.random.randint(0, init_img["size"], init_img["centers"])
        y_coord = np.random.randint(0, init_img["size"], init_img["centers"])

        for index in range(len(x_coord)):
            # Create an outer and inner circle. Then subtract the inner from the outer.
            radius = 1
            ri, ci = draw.circle(x_coord[index], y_coord[index], radius=radius, shape=data.shape)
            data[ri, ci] = 1

    copy_image = deepcopy(data)

    # generate inverse data.
    copy_image[np.where(data == 0)] = N_max
    copy_image[np.where(data == 1)] = 0

    return copy_image


def get_local_mask(pass_time):
    coord_list = np.array(list(range(7))) - 3
    all_mask = np.array(list(product(coord_list, coord_list)))

    assert pass_time == 1 or pass_time == 2, "'pass_time' should be 1 or 2."

    if pass_time == 1:
        local_mask_maps = all_mask[np.where((all_mask[:, 0] < 0) | ((all_mask[:, 0] == 0) & (all_mask[:, 1] < 0)))]
    elif pass_time == 2:
        local_mask_maps = all_mask[np.where((all_mask[:, 0] > 0) | ((all_mask[:, 0] == 0) & (all_mask[:, 1] > 0)))]

    return local_mask_maps


def first_pass(sel_distance, sel_image):
    height = sel_image.shape[0]
    width = sel_image.shape[1]

    copy_img = deepcopy(sel_image)

    for _y in range(height):
        for _x in range(width):
            local_masks = []
            for local_co in get_local_mask(1):
                # local_co is coordinate of neighborhood.
                try:
                    local_mask = sel_distance(_y, _y + local_co[1], _x, _x + local_co[0]) + \
                                 copy_img[_y + local_co[1]:_y + local_co[1] + 1, _x + local_co[0]:_x + local_co[0] + 1]
                    if len(local_mask) == 0:
                        local_mask = N_max
                except:
                    local_mask = N_max

                local_masks.append(local_mask)

            # add center.
            local_masks.append(copy_img[_y, _x])

            try:
                copy_img[_y, _x] = np.min(local_masks).item()
            except:
                copy_img[_y, _x] = np.min(local_masks)

    return copy_img


def second_pass(sel_distance, sel_image):
    height = sel_image.shape[0]
    width = sel_image.shape[1]

    copy_img = deepcopy(sel_image)

    for _y in range(height - 1, -1, -1):
        for _x in range(width - 1, -1, -1):
            local_masks = []
            for local_co in get_local_mask(2):
                # local_co is coordinate of neighborhood.
                try:
                    local_mask = sel_distance(_y, _y + local_co[1], _x, _x + local_co[0]) + \
                                 copy_img[_y + local_co[1]:_y + local_co[1] + 1, _x + local_co[0]:_x + local_co[0] + 1]
                    if len(local_mask) == 0:
                        local_mask = N_max
                except:
                    local_mask = N_max

                local_masks.append(local_mask)

            # add center.
            local_masks.append(copy_img[_y, _x])

            try:
                copy_img[_y, _x] = np.min(local_masks).item()
            except:
                copy_img[_y, _x] = np.min(local_masks)

    return copy_img

# main script ----------------------------------------------------------------------

# # # 1. Normal sample.---------------------------------------------
# # generate random centers.
# se_type = "D_4"
#
# image = generate_image()
#
# middle_result = image
#
# is_repeat = True
# while is_repeat:
#     result_AL = first_pass(distance_type["D_E"], middle_result)
#     final_result = second_pass(distance_type["D_E"], result_AL)
#
#     if np.array_equal(final_result, middle_result):
#         is_repeat = False
#     else:
#         print("repeat")
#         middle_result = final_result
#
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# ax = plt.gca()
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         ax.text(x, y, str(image[y, x])[:3], color='white', ha='center', va='center')
# plt.title("Sample image")
#
# plt.subplot(1, 2, 2)
# plt.imshow(final_result)
# ax = plt.gca()
# for y in range(final_result.shape[0]):
#     for x in range(final_result.shape[1]):
#         ax.text(x, y, str(final_result[y, x])[:3], color='white', ha='center', va='center')
#
# plt.title("Sample image with {}".format(se_type))
# plt.show()

# # # 1. DE sample. ---------------------------------------------
image_option = {"centers": 3, "size": 64}

# generate random centers.
image = generate_image(use_sample=False, init_img=image_option)

for se_type in distance_type:
    middle_result = image

    is_repeat = True
    while is_repeat:
        result_AL = first_pass(distance_type[se_type], middle_result)
        final_result = second_pass(distance_type[se_type], result_AL)

        if np.array_equal(final_result, middle_result):
            is_repeat = False
        else:
            print("repeat")
            middle_result = final_result

    X, Y = np.meshgrid(list(range(image.shape[0])), list(range(image.shape[1])))

    plt.subplot(1, 3, 1)
    plt.title("Random ones")
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.title("Random ones")
    plt.imshow(result_AL)
    plt.subplot(1, 3, 3)
    plt.imshow(final_result)
    plt.contour(X, Y, final_result[Y, X], 12, colors='white', linewidths=.2)
    plt.title("Random ones with {}".format(se_type))
    plt.show()
