import numpy as np
import cv2
from skimage.util import random_noise
import random
from glob import glob
from tqdm import tqdm
import os
import shutil

def distance(a, b):
  return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def add_noise(img, r_c, c_c, color=255):
    row, col, _ = img.shape
    d = min(row, col)//1
    filter = img.copy().astype(np.uint32)
    for i in range(row):
        for j in range(col):
            filter[i, j, :] = np.exp(-distance((i, j), (r_c, c_c))**2/(2*d**2))*color
    img = img+filter
    img[np.where(img>255)] = 255
    return img


def flip_image(image, raito=2, kind="tl"):
    h = int(image.shape[0]*raito)
    w = int(image.shape[1]*raito)
    img_flip = np.zeros((h, w, 3), dtype=np.uint8)

    if kind=="tl":
        img_flip[h-image.shape[0]:, w-image.shape[1]:] = image.copy()
    elif kind=="br":
        img_flip[:image.shape[0], :image.shape[1]] = image.copy()
    elif kind=="tr":
        img_flip[h-image.shape[0]:, :image.shape[1]] = image.copy()
    elif kind=="bl":
        img_flip[:image.shape[0], w-image.shape[1]:] = image.copy()

    return img_flip

def crop_initalize_image(image, image_flip, kind="tl"):
    h, w, _ = image_flip.shape
    if kind=="tl":
        img = image_flip[h-image.shape[0]:, w-image.shape[1]:]
    elif kind=="br":
        img = image_flip[:image.shape[0], :image.shape[1]]
    elif kind=="tr":
        img = image_flip[h-image.shape[0]:, :image.shape[1]]
    elif kind=="bl":
        img = image_flip[:image.shape[0], w-image.shape[1]:]

    return img


def add_tint(image, color, kind):
    img = image.copy()

    img = flip_image(img, 1.2, kind="tl")

    if kind=="tl":
        img = add_noise(img, 0, 0, color=color)
    elif kind=="br":
        img = add_noise(img, img.shape[0], img.shape[1], color=color)
    elif kind=="tr":
        img = add_noise(img, 0, img.shape[1], color=color)
    elif kind=="bl":
        img = add_noise(img, img.shape[0], 0, color=color)


    img = img.astype(np.uint8)

    img = crop_initalize_image(image, img, kind="tl")
    return img


def add_tint_full_image(image):
    p_value = random.randint(20, 30)
    tint_mask = np.ones_like(image)*(p_value, 0, p_value)
    tint_img = image+tint_mask
    tint_img[tint_img>255]=255
    tint_img = tint_img.astype(np.uint8)
    return tint_img


def get_random_color_value(kind):
    if kind=='r':
        color = np.array([0, 0, random.randint(20, 50)], dtype=np.uint8)
    elif kind=='y':
        y_value = random.randint(20, 50)
        color = np.array([0, y_value, y_value], dtype=np.uint8)
    elif kind=='g':
        color = np.array([0, random.randint(20, 50), 0], dtype=np.uint8)

    return color


def random_spot(image, color):
    img = image.copy()
    if color=='r':
        r_value = random.randint(128, 255)
        color_value = np.array([0, 0, r_value], dtype=np.uint8)
    elif color=='y':
        y_value = random.randint(128, 255)
        color_value = np.array([0, y_value, y_value], dtype=np.uint8)        
    elif color=='g':
        g_value = random.randint(128, 255)
        color_value = np.array([0, g_value, 0], dtype=np.uint8)
    elif color=='w':
        color_value = np.array([255, 255, 255], dtype=np.uint8)   

    row, col, _ = img.shape
    d = min(row, col)//10
    filter = img.copy().astype(np.uint32)
    random_rc = list(range(0, img.shape[0]))
    # del random_rc[int(img.shape[0]/5): int(4*img.shape[0]/5)]
    if len(random_rc[int(img.shape[0]/5): int(4*img.shape[0]/5)]) != 0:
        random_rc = random_rc[int(img.shape[0]/5): int(4*img.shape[0]/5)]
    random_cc = list(range(0, img.shape[1]))
    # del random_cc[int(img.shape[1]/5): int(4*img.shape[1]/5)]
    if len(random_cc[int(img.shape[1]/5): int(4*img.shape[1]/5)]) != 0:
        random_cc = random_cc[int(img.shape[1]/5): int(4*img.shape[1]/5)]
    r_c = random.choice(random_rc)
    c_c = random.choice(random_cc)
    for i in range(row):
        for j in range(col):
            filter[i, j, :] = np.exp(-distance((i, j), (r_c, c_c))**2/(2*d**2))*color_value
    img = img+filter
    img[np.where(img>255)] = 255
    img = img.astype(np.uint8)
    return img


if __name__ == "__main__":
    # lb = 'r'
    # k = 1006
    # time = "ori"
    # img_path_list = glob(f"/home/phuongdoan/Desktop/data_TFL/{lb}/*.jpg")
    # img_path_list = random.sample(img_path_list, k=k)
    # path_save = "/home/phuongdoan/Desktop/Generate"

    # for path in tqdm(img_path_list):
    #     img = cv2.imread(path)
        # img = add_tint(img, get_random_color_value(kind='g'), "tr")
        # img = random_spot(img, color='w')
        # img = random_spot(img, color='g')
        # img = add_tint_full_image(img)
        # cv2.imwrite(f"{path_save}/{lb}/generate_{time}_{path.split('/')[-1]}", img)

    # img_path_list = glob(f"/home/phuongdoan/Desktop/data_TFL/n/*.jpg")
    # for path in img_path_list:
    #     img = cv2.imread(path)
    #     h, w, c = img.shape
    #     if h/w < 2.5:
    #         os.remove(path)


    # img = cv2.imread("/home/phuongdoan/Desktop/data_TFL/000/000132.jpg")
    # img = add_tint(img, get_random_color_value(kind='y'), "tl")
    # img = add_tint_full_image(img)
    # img = random_spot(img, color='r')
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_path_list = glob("/home/phuongdoan/Desktop/Generate/g/*.jpg")
    img_path_list = random.sample(img_path_list, k=13000)
    path_save = "/home/phuongdoan/Desktop/Data_TFL/train/g"
    path_val = "/home/phuongdoan/Desktop/Data_TFL/val/y"

    for path in tqdm(img_path_list):
        # img = cv2.imread(path)
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imwrite(f"{path_save}/rotate_{path.split('/')[-1]}", img)
        shutil.move(path, f"{path_save}/{path.split('/')[-1]}")