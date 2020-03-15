# -*- coding:utf-8 -*-
import PIL.Image as Image
import os
import cv2
import numpy as np

IMAGES_PATH = './after_test/'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
IMAGE_SAVE_PATH = 'final.jpg'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]


# 定义图像拼接函数
def image_compose(h, w, IMAGE_ROW, IMAGE_COLUMN):
    print(IMAGE_ROW * h, IMAGE_COLUMN * w)
    to_image = Image.new('RGB', (IMAGE_COLUMN * w, IMAGE_ROW * h))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (h, w), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * h, (y - 1) * w))
    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


def image_compose2(h, w, IMAGE_ROW, IMAGE_COLUMN):
    to_image = np.ndarray([IMAGE_ROW * h,IMAGE_COLUMN * w,3],np.uint8)
    print(to_image.shape)
    for y in range(IMAGE_ROW):
        for x in range(IMAGE_COLUMN):
            start_x =x*w
            end_x = (x+1)*w
            start_y = y*h
            end_y = (y+1)*h
            img_path = IMAGES_PATH+str(x).zfill(3)+"_"+str(y).zfill(3)+".jpg"
            tmp_block = cv2.imread(img_path)
            to_image[start_y:end_y,start_x:end_x,:]=tmp_block
    cv2.imwrite("out2.tif", to_image)