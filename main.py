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
    distance = np.sqrt(np.square(xys[0] - xys[1]) + np.square(xys[2] - xys[3]))
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
    if np.abs(xys[0] - xys[1]) > np.abs(xys[2] - xys[3]):
        distance = np.abs(xys[0] - xys[1]) + (np.sqrt(2) - 1) * np.abs(xys[2] - xys[3])
    else:
        distance = (np.sqrt(2) - 1) * np.abs(xys[0] - xys[1]) + np.abs(xys[2] - xys[3])
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

    # sample = np.array([[0, 0, 0, 0, 0, 0, 1, 0],
    #                    [0, 0, 0, 0, 0, 1, 0, 0],
    #                    [0, 0, 0, 0, 0, 1, 0, 0],
    #                    [0, 0, 0, 0, 0, 1, 0, 0],
    #                    [0, 1, 1, 0, 0, 0, 1, 0],
    #                    [0, 1, 0, 0, 0, 0, 0, 1],
    #                    [0, 1, 0, 0, 0, 0, 0, 0],
    #                    [0, 1, 0, 0, 0, 0, 0, 0]
    #                    ], dtype=np.float)

    sample = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       ], dtype=np.float)
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


def calculate_distance_al(sel_distance, sel_image):
    height = sel_image.shape[0]
    width = sel_image.shape[1]

    copy_img = deepcopy(sel_image)

    for y in range(height):
        for x in range(width):
            try:
                local_mask_1 = sel_distance(y, y - 1, x, x - 1) + copy_img[y - 1:y, x - 1:x]
                if len(local_mask_1) == 0:
                    local_mask_1 = N_max
            except:
                local_mask_1 = N_max
            try:
                local_mask_2 = sel_distance(y, y - 1, x, x) + copy_img[y - 1:y, x:x + 1]
                if len(local_mask_2) == 0:
                    local_mask_2 = N_max
            except:
                local_mask_2 = N_max
            try:
                local_mask_3 = sel_distance(y, y, x, x - 1) + copy_img[y:y + 1, x - 1:x]
                if len(local_mask_3) == 0:
                    local_mask_3 = N_max
            except:
                local_mask_3 = N_max
            try:
                local_mask_4 = sel_distance(y, y + 1, x, x - 1) + copy_img[y + 1:y + 2, x - 1:x]
                if len(local_mask_4) == 0:
                    local_mask_4 = N_max
            except:
                local_mask_4 = N_max

            local_mask_center = copy_img[y, x]

            try:
                copy_img[y, x] = np.min(
                    [local_mask_1, local_mask_2, local_mask_3, local_mask_4, local_mask_center]).item()
            except:
                copy_img[y, x] = np.min([local_mask_1, local_mask_2, local_mask_3, local_mask_4, local_mask_center])

    return copy_img


def calculate_distance_br(sel_distance, sel_image):
    height = sel_image.shape[0]
    width = sel_image.shape[1]

    copy_img = deepcopy(sel_image)

    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            try:
                local_mask_1 = sel_distance(y, y + 1, x, x + 1) + copy_img[y + 1:y + 2, x + 1:x + 2]
                if len(local_mask_1) == 0:
                    local_mask_1 = N_max
            except:
                local_mask_1 = N_max
            try:
                local_mask_2 = sel_distance(y, y + 1, x, x) + copy_img[y + 1:y + 2, x:x + 1]
                if len(local_mask_2) == 0:
                    local_mask_2 = N_max
            except:
                local_mask_2 = N_max
            try:
                local_mask_3 = sel_distance(y, y, x, x + 1) + copy_img[y:y + 1, x + 1: x + 2]
                if len(local_mask_3) == 0:
                    local_mask_3 = N_max
            except:
                local_mask_3 = N_max
            try:
                local_mask_4 = sel_distance(y, y - 1, x, x + 1) + copy_img[y - 1:y, x + 1:x + 2]
                if len(local_mask_4) == 0:
                    local_mask_4 = N_max
            except:
                local_mask_4 = N_max

            local_mask_center = copy_img[y, x]

            try:
                copy_img[y, x] = np.min(
                    [local_mask_1, local_mask_2, local_mask_3, local_mask_4, local_mask_center]).item()
            except:
                copy_img[y, x] = np.min([local_mask_1, local_mask_2, local_mask_3, local_mask_4, local_mask_center])

    return copy_img


# main script ----------------------------------------------------------------------

# # 1. Normal sample.---------------------------------------------
# generate random centers.
image = generate_image()
result_AL = calculate_distance_al(distance_type["D_QE"], image)
final_result = calculate_distance_br(distance_type["D_QE"], result_AL)

result_AL = calculate_distance_al(distance_type["D_QE"], final_result)
final_result = calculate_distance_br(distance_type["D_QE"], result_AL)

result_AL = calculate_distance_al(distance_type["D_QE"], final_result)
final_result = calculate_distance_br(distance_type["D_QE"], result_AL)

plt.subplot(1, 2, 1)
plt.imshow(image)
ax = plt.gca()
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        ax.text(x, y, str(image[y, x])[:3], color='white', ha='center', va='center')
plt.title("Sample image")

plt.subplot(1, 2, 2)
plt.imshow(final_result)
ax = plt.gca()
for y in range(final_result.shape[0]):
    for x in range(final_result.shape[1]):
        ax.text(x, y, str(final_result[y, x])[:3], color='white', ha='center', va='center')

# plt.title("Sample image with D_E")
plt.show()

# # # 1. DE sample. ---------------------------------------------
#
# iterations = 60
#
# image_option = {"centers": 3, "size": 16}
#
# # generate random centers.
# image = generate_image(use_sample=False, init_img=image_option)
#
#
# result_AL = calculate_distance_al(distance_type["D_E"], image)
# final_result = calculate_distance_br(distance_type["D_E"], result_AL)
#
# for i in range(iterations):
#     result_AL = calculate_distance_al(distance_type["D_E"], final_result)
#     final_result = calculate_distance_br(distance_type["D_E"], result_AL)
#
# X, Y = np.meshgrid(list(range(image.shape[0])), list(range(image.shape[1])))
#
# plt.subplot(1, 3, 1)
# plt.title("Random ones")
# plt.imshow(image)
# plt.subplot(1, 3, 2)
# plt.title("Random ones")
# plt.imshow(result_AL)
# plt.subplot(1, 3, 3)
# plt.imshow(final_result)
# plt.contour(X, Y, final_result[Y, X], 12, colors='white', linewidths=.2)
# plt.title("Random ones with D_E")
# plt.show()
#
# # generate random centers.
# # image = generate_image(use_sample=False, init_img=image_option)
# result_AL = calculate_distance_al(distance_type["D_4"], image)
# final_result = calculate_distance_br(distance_type["D_4"], result_AL)
# for i in range(iterations):
#     result_AL = calculate_distance_al(distance_type["D_4"], final_result)
#     final_result = calculate_distance_br(distance_type["D_4"], result_AL)
#
# X, Y = np.meshgrid(list(range(image.shape[0])), list(range(image.shape[1])))
#
# plt.subplot(1, 3, 1)
# plt.title("Random ones")
# plt.imshow(image)
# plt.subplot(1, 3, 2)
# plt.title("Random ones")
# plt.imshow(result_AL)
# plt.subplot(1, 3, 3)
# plt.imshow(final_result)
# plt.contour(X, Y, final_result[Y, X], 8, colors='white', linewidths=.2)
# plt.title("Random ones with D_4")
# plt.show()
#
# # generate random centers.
# # image = generate_image(use_sample=False, init_img=image_option)
# result_AL = calculate_distance_al(distance_type["D_8"], image)
# final_result = calculate_distance_br(distance_type["D_8"], result_AL)
# for i in range(iterations):
#     result_AL = calculate_distance_al(distance_type["D_8"], final_result)
#     final_result = calculate_distance_br(distance_type["D_8"], result_AL)
#
# X, Y = np.meshgrid(list(range(image.shape[0])), list(range(image.shape[1])))
#
# plt.subplot(1, 3, 1)
# plt.title("Random ones")
# plt.imshow(image)
# plt.subplot(1, 3, 2)
# plt.title("Random ones")
# plt.imshow(result_AL)
# plt.subplot(1, 3, 3)
# plt.imshow(final_result)
# plt.contour(X, Y, final_result[Y, X], 8, colors='white', linewidths=.2)
# plt.title("Random ones with D_8 ")
# plt.show()
#
# # generate random centers.
# # image = generate_image(use_sample=False, init_img=image_option)
# result_AL = calculate_distance_al(distance_type["D_QE"], image)
# final_result = calculate_distance_br(distance_type["D_QE"], result_AL)
#
# for i in range(iterations):
#     result_AL = calculate_distance_al(distance_type["D_QE"], final_result)
#     final_result = calculate_distance_br(distance_type["D_QE"], result_AL)
#
# X, Y = np.meshgrid(list(range(image.shape[0])), list(range(image.shape[1])))
#
# plt.subplot(1, 3, 1)
# plt.title("Random ones")
# plt.imshow(image)
# plt.subplot(1, 3, 2)
# plt.title("Random ones")
# plt.imshow(result_AL)
# plt.subplot(1, 3, 3)
# plt.imshow(final_result)
# plt.contour(X, Y, final_result[Y, X], 8, colors='white', linewidths=.2)
# plt.title("Random ones with D_QE")
# plt.show()
