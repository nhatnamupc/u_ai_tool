import os
import shutil

import cv2
import numpy as np
from PIL import Image
import random
import threading
import csv

from rembg import remove

join = os.path.join
basename = os.path.basename
ROOT = r"\\192.168.0.241\nam\yakult_project"
ROOT1 = r"/home/upc/WorkSpaces/nam/yakult_project/"
IMAGE_PROCESS_PATH = r"/home/upc/WorkSpaces/nam/yakult_project/images_processed"


def get_all_file(path_dir):  # Get all file and folder with links from path_dir w
    file_list, dir_list = [], []
    for rdir, subdir, files in os.walk(path_dir):
        file_list.extend([os.path.join(rdir, f) for f in files])
        dir_list.extend([os.path.join(rdir, d) for d in subdir])
    return file_list, dir_list


def remove_background(image, save_path=None):
    """Input : Path of image => Output : removed background image by OpenCV"""
    in_ = cv2.imread(image)
    out_ = remove(in_)
    if save_path is not None:
        cv2.imwrite(save_path, out_)
    else:
        return out_


def toImgOpenCV(imgPIL):  # Converse imgPIL to imgOpenCV
    i = np.array(imgPIL)  # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i


def toImgPIL(imgOpenCV): return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))


def w_h_image_rotate(image):
    im = toImgOpenCV(image)
    im_to_crop = toImgOpenCV(image)

    alpha_channel = im[:, :, 3]
    rgb_channel = im[:, :, :3]
    white_background = np.ones_like(rgb_channel, dtype=np.uint8) * 255

    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    base = rgb_channel.astype(np.float32) * alpha_factor
    white = white_background.astype(np.float32) * (1 - alpha_factor)
    final_im = base + white
    final_im = final_im.astype(np.uint8)

    gray = cv2.cvtColor(final_im, cv2.COLOR_BGR2GRAY)
    r1, t1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # t1=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    c1, h1 = cv2.findContours(t1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(c1, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnt[0])
    return x, y, (x + w), (y + h)


def cut_from_removed_background(src_, save_path=None):
    files_, _ = get_all_file(src_)
    for image in files_:
        im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        im_to_crop = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        alpha_channel = im[:, :, 3]
        rgb_channel = im[:, :, :3]
        white_background = np.ones_like(rgb_channel, dtype=np.uint8) * 255

        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

        base = rgb_channel.astype(np.float32) * alpha_factor
        white = white_background.astype(np.float32) * (1 - alpha_factor)
        final_im = base + white
        final_im = final_im.astype(np.uint8)

        gray = cv2.cvtColor(final_im, cv2.COLOR_BGR2GRAY)
        r1, t1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # t1=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        c1, h1 = cv2.findContours(t1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cnt = sorted(c1, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(cnt[0])
        crop = im_to_crop[y:y + h, x:x + w]
        cv2.imwrite(join(save_path,basename(image)), crop)

        # if resize_value1 > 0:
    #     crop_rz_1 = cv2.resize(crop, (0, 0), fx=resize_value1, fy=resize_value1)
    #     cv2.imwrite(save[:-4] + '-' + str(resize_value1) + PNG, crop_rz_1)
    #
    # if resize_value2 > 0:
    #     crop_rz_2 = cv2.resize(crop, (0, 0), fx=resize_value2, fy=resize_value2)
    #     cv2.imwrite(save[:-4] + '-' + str(resize_value2) + PNG, crop_rz_2)


def rotated():
    pass


def class_id(file):
    files = file.split("_")
    return str(files[0])


def bnd_box_to_yolo_line(box, img_size):
    (x_min, y_min) = (box[0], box[1])
    (w, h) = (box[2], box[3])
    x_max = x_min + w
    y_max = y_min + h

    x_center = float((x_min + x_max)) / 2 / img_size[1]
    y_center = float((y_min + y_max)) / 2 / img_size[0]

    w = float((x_max - x_min)) / img_size[1]
    h = float((y_max - y_min)) / img_size[0]

    return x_center, y_center, w if w < 1 else 1.0, h if h < 1 else 1.0


def merge(path_background, path_foreground, annotation_num, save_path, rotate=False, filename="merged"):
    """Paste foreground to background """

    backgrounds, _ = get_all_file(path_background)  # Get list backgrounds
    foregrounds, _ = get_all_file(path_foreground)  # Get list foregrounds
    dictionary = {}  # ex: {"1": [1,0]}  => [1,0] : 1 annotation value, 0 rotate degree value
    overlap_value = 3  # Overlap value
    idx = 0  # Image index
    name = "no_name"
    while True:
        if len(foregrounds) == 0:  # Break when len of list foreground == 0
            break

        # Random choice one of background in list backgrounds with 4 channel
        bg = Image.open(random.choice(backgrounds)).convert("RGBA")
        merged_image = bg.copy()
        bw, bh = merged_image.size  # Get weight height of merge image
        cur_h, cur_w, max_h, max_w = 0, 0, 0, 0
        is_write = False
        while True:
            if len(foregrounds) == 0:
                break
            fg = random.choice(foregrounds)  # Random choice foreground
            id_ = class_id(basename(fg))
            if id_ not in dictionary:
                dictionary[id_] = [1, 0]
            else:
                if dictionary[id_][0] > annotation_num:
                    foregrounds.remove(fg)
                    continue
            fore_image = Image.open(fg).convert("RGBA")
            if rotate:
                if dictionary[id_][1] > 359:
                    dictionary[id_][1] = 0
                dictionary[id_][1] += 20
                fore_image = fore_image.rotate(dictionary[id_][1], expand=True)
                fore_image = fore_image.crop(w_h_image_rotate(fore_image))
            fw, fh = fore_image.size  # Background size
            if fw > bw or fh > bh:
                continue

            if max_h < fh:
                max_h = fh - overlap_value
            if (cur_w + fw) >= bw:
                cur_w = 0
                cur_h += max_h
            if (cur_h + fh) >= bh:
                break
            x, y = 0, 0
            try:
                if cur_w > 0:
                    if cur_h == 0:
                        merged_image.paste(fore_image, (cur_w - overlap_value, cur_h), fore_image)
                        x, y = cur_w - overlap_value, cur_h
                    else:
                        merged_image.paste(fore_image, (cur_w, cur_h - overlap_value), fore_image)
                        x, y = cur_w, cur_h - overlap_value
                else:
                    merged_image.paste(fore_image, (cur_w, cur_h), fore_image)
                    x, y = cur_w, cur_h
            except Exception as e:
                print(e)
            box = (x, y, fw, fh)
            yolo_box = bnd_box_to_yolo_line(box, (bh, bw))
            cls = int(id_)
            name = join(save_path, f"{filename}_{idx}")
            with open(name + ".txt", 'a') as f:
                f.write(f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
            is_write = True
            dictionary[id_][0] += 1
            cur_w += fw - overlap_value

        if is_write:
            merged_image.save(name + ".png", format="png")

        idx += 1


def get_frames(src_: str, save_, value=10):
    if src_.endswith("png"):
        folder_id = join(save_, class_id(basename(src_)))
        os.makedirs(folder_id, exist_ok=True)
        old, new = src_, join(folder_id, basename(src_))
        shutil.copy(old, new)
    elif src_.endswith("mp4") or src_.endswith("avi"):
        cap = cv2.VideoCapture(src_)
        stt, index = 1, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            folder_id = join(save_, class_id(basename(src_)))
            os.makedirs(folder_id, exist_ok=True)
            if index % value == 0:
                cv2.imwrite(join(folder_id, basename(src_)[:-4] + f"{stt}.png"), frame)
            stt += 1
            index += 1


def count_annotation(path_txt, file_classes_name, write_to_csv=False):
    annotation_dic = {}
    f_name = open(file_classes_name, 'r')
    d_name = f_name.readlines()
    f_name.close()
    names = []
    for d in d_name:
        names.append(d.rstrip('\n'))

    files, _ = get_all_file(path_txt)
    for txt in files:
        if txt.endswith('txt'):
            f1 = open(txt, 'r')
            data = f1.readlines()
            f1.close()
            for dt in data:
                cls, x, y, w, h = map(float, dt.split(' '))
                if cls not in annotation_dic.keys():
                    annotation_dic[int(cls)] = 1
                else:

                    annotation_dic[int(cls)] += 1

        for index, id_cls in enumerate(names):
            if index not in annotation_dic.keys():
                annotation_dic[index] = 0

        if write_to_csv:
            header = names
            data = []
            for k, v in sorted(annotation_dic.items()):
                data.append(v)

            with open('../utils/test.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)


def remove_background_thread(src, save_):
    files_, _ = get_all_file(src)
    for img in files_:
        save_f = join(save_, class_id(basename(img)))
        os.makedirs(save_f, exist_ok=True)
        remove_background(img, join(save_f, basename(img)))


# Get Frame
# files, _ = get_all_file(join(ROOT1, "images_processed", "side_process", "backside", "video"))
# save = join(ROOT1, "images_processed", "side_process", "backside","extracted")
# for f in files:
#     get_frames(f, save, value=10)

# Remove background
# files, _ = get_all_file(join(ROOT1, "images_processed", "side_process", "backside", "extracted"))
#
# save = join(ROOT1, "images_processed", "side_process", "backside", "removed")
# for f in os.listdir(join(ROOT1, "images_processed", "side_process", "backside", "extracted")):
#     # remove_background_thread(f,save)
#     if f in ["13"]:
#         t = threading.Thread(target=remove_background_thread, args=[join(ROOT1, "images_processed", "side_process", "backside", "extracted",f), save])
#         t.start()
#

# Cut and process
#
# save = join(ROOT1, "images_processed", "side_process", "backside", "processed")
# src = join(ROOT1, "images_processed", "side_process", "backside", "removed")
# for f in os.listdir(src):
#     if f in ["13"]:
#         t = threading.Thread(target=cut_from_removed_background,
#                              args=[join(ROOT1, "images_processed", "side_process", "backside", "removed", f), save])
#         t.start()

# Merge function
# SAVE_PATH = r"\\192.168.0.241\nam\yakult_project\images_processed\top_removed"

# images, _ = get_all_file(join(IMAGE_PROCESS_PATH, "1"))
save = join(IMAGE_PROCESS_PATH,"side_process", "backside","merged")
fore_path = join(IMAGE_PROCESS_PATH, "side_process", "backside","processed")
back_path = join(IMAGE_PROCESS_PATH, "background")

# for index in range(0, 50, 5):
#     t = threading.Thread(target=merge, args=[back_path, fore_path, 5, save, index])
#     t.start()
merge(back_path, fore_path, 100, save, rotate=False, filename="merged_backside")
# count_annotation(save, "../utils/name.txt", write_to_csv=True)

# for img in images:
# t = threading.Thread(target=remove_background, args=[img, join(SAVE_PATH, basename(img))])
# t.start()
# t = threading.Thread(target=cut_from_removed_background, args=[img, join(SAVE_PATH, basename(img))])
# t.start()
