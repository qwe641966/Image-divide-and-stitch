# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2
import concact


def divide_method1(img, w, h, m, n):  # 分割成m行n列
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = np.round(gx).astype(np.int)
    gy = np.round(gy).astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, int(h * 1.0 / (m - 1) + 0.5), int(w * 1.0 / (n - 1) + 0.5), 3],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, 0:gy[i + 1][j + 1] - gy[i][j], 0:gx[i + 1][j + 1] - gx[i][j], :] = img[
                                                                                                  gy[i][j]:gy[i + 1][
                                                                                                      j + 1],
                                                                                                  gx[i][j]:gx[i + 1][
                                                                                                      j + 1],
                                                                                                  :]  # 这样写比a[i,j,...]=要麻烦，但是可以避免网格分块的时候，有些图像块的比其他图像块大一点或者小一点的情况引起程序出错
    return divide_image


def divide_method2(img, w, h, m, n):  # 分割成m行n列
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)  # 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]  #
    return divide_image


def display_blocks(divide_image):  #

    m, n = divide_image.shape[0], divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(divide_image[i, j, :])
            plt.axis('off')
            plt.title('block:' + str(i * n + j + 1))
            # savejpg_name = str(j + i*n).zfill(4) + '.jpg'  # 拼接保存图像的地址
            savejpg_name = str(j).zfill(3)+"_"+str(i).zfill(3)+".jpg"
            im_path = 'output'+'/'+savejpg_name
            cv2.imwrite(im_path, divide_image[i, j, :])   # 保存图像

def image_concat(divide_image):
    m, n, grid_h, grid_w = [divide_image.shape[0], divide_image.shape[1],  # 每行，每列的图像块数
                            divide_image.shape[2], divide_image.shape[3]]  # 每个图像块的尺寸

    restore_image = np.zeros([m * grid_h, n * grid_w, 3], np.uint8)

    restore_image[0:grid_h, 0:]
    for i in range(m):
        for j in range(n):
            restore_image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w] = divide_image[i, j, :]
    return restore_image


def main():
    # 读取原始图片
    img = cv2.imread('rs.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('\t\t\t  原始图像形状:\n', '\t\t\t', img.shape)
    h, w = img.shape[0], img.shape[1]
    fig1 = plt.figure('原始图像')  # fig
    plt.imshow(img)

    plt.axis('off')
    plt.title('Original image')
    plt.show()
    # 原始图像分块
    m = 29
    n = 33

    block_h = int(h/m)
    block_w = int(w/n)

    ###########################四舍五入#####################################
    # divide_image1 = divide_method1(img, w, h, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
    # fig2 = plt.figure('分块后的子图像:四舍五入法')
    # display_blocks(divide_image1)

    ###########################图像缩放法#####################################
    # divide_image2 = divide_method2(img, w, h, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
    # fig3 = plt.figure('分块后的子图像：图像缩放法')
    # display_blocks(divide_image2)

    ###########################分块图像还原###################################
    #fig4 = plt.figure('分块图像的还原')
    #restore_image1 = image_concat(divide_image1)  # 四舍五入法分块还原
    #restore_image2 = image_concat(divide_image2)  # 图像缩放法分块还原
    #plt.subplot(1, 2, 1)
    #plt.imshow(restore_image1)
    #plt.axis('off')
    #plt.title('Rounding')
    #plt.subplot(1, 2, 2)
    #plt.imshow(restore_image2)
    #plt.axis('off')
    #plt.title('Scaling')
    #print('\n\t\t\t还原后的图像尺寸')
    #print('\t''‘图像缩放法’：', restore_image2.shape)
    #print('\t‘四舍五入法’：', restore_image1.shape, '\t''‘图像缩放法’：', restore_image2.shape)
    #plt.show()

    ###########################图像拼接算法(非原始)###################################
    concact.image_compose2(block_h, block_w, m, n)

if __name__ == "__main__":
    main()