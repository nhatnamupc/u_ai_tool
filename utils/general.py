import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import threading
from utils import images_process as u
from datetime import datetime

import csv

from rembg import remove
import albumentations as A

join = os.path.join
basename = os.path.basename
ROOT = r"\\192.168.0.241\nam\yakult_project"
IMAGE_PROCESS_PATH = r"\\192.168.0.241\nam\yakult_project\images_processed"


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


def cut_from_removed_background(image, save_path=None):
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
    cv2.imwrite(save_path, crop)

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


def yolo_box_to_rec_box(box, img_size):
    x, y, w, h = box
    x1 = int((x - w / 2) * img_size[1])
    w1 = int((x + w / 2) * img_size[1])
    y1 = int((y - h / 2) * img_size[0])
    h1 = int((y + h / 2) * img_size[0])
    if x1 < 0:
        x1 = 0
    if w1 > img_size[1] - 1:
        w1 = img_size[1] - 1
    if y1 < 0:
        y1 = 0
    if h1 > img_size[0] - 1:
        h1 = img_size[0] - 1
    return x1, y1, w1 - x1, h1 - y1


def bnd_box_to_yolo_box(box, img_size):
    (x_min, y_min) = (box[0], box[1])
    (w, h) = (box[2], box[3])
    x_max = x_min + w
    y_max = y_min + h

    x_center = float((x_min + x_max)) / 2 / img_size[1]
    y_center = float((y_min + y_max)) / 2 / img_size[0]

    w = float((x_max - x_min)) / img_size[1]
    h = float((y_max - y_min)) / img_size[0]

    return x_center, y_center, w if w < 1 else 1.0, h if h < 1 else 1.0


def merge(path_background, path_foreground, annotation_num, save_path, rotate=False, filename="merged_2"):
    """Paste foreground to background """

    backgrounds, _ = get_all_file(path_background)  # Get list backgrounds
    foregrounds, _ = get_all_file(path_foreground)  # Get list foregrounds
    dictionary = {}  # ex: {"1": [1,0]}  => [1,0] : 1 annotation value, 0 rotate degree value
    overlap_value = 10  # Overlap value
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
                    print(f"{id_} finnish")
                    foregrounds.remove(fg)
                    continue
            fore_image = Image.open(fg).convert("RGBA")  # Read foreground image with 4 channels
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
            yolo_box = bnd_box_to_yolo_box(box, (bh, bw))  # Converse Bounding Box(xywh) to Yolo format(xyxy)
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


def get_frames(src: str, save_, value=10):
    if src.endswith("png"):
        folder_id = join(save_, class_id(basename(src)))
        os.makedirs(folder_id, exist_ok=True)
        old, new = src, join(folder_id, basename(src))
        shutil.copy(old, new)
    elif src.endswith("mp4") or src.endswith("avi"):
        cap = cv2.VideoCapture(src)
        stt, index = 1, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            folder_id = join(save_, class_id(basename(src)))
            os.makedirs(folder_id, exist_ok=True)
            if index % value == 0:
                cv2.imwrite(join(folder_id, basename(src)[:-4] + f"{stt}.png"), frame)
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
                if cls in annotation_dic.keys():
                    annotation_dic[cls] += 1
                else:
                    annotation_dic[cls] = 1

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
    print(annotation_dic)


def remove_background_thread(src, save_):
    save_f = join(save_, class_id(basename(src)))
    os.makedirs(save_f, exist_ok=True)
    remove_background(src, join(save_f, basename(src)))


def contras_brightness(src, save_light, save_dark, plus_alpha=1.6, minus_alpha=0.8):
    original_images, _ = get_all_file(src)
    alpha = 1.5
    for image in original_images:
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        plus_contras, minus_contrast = cv2.convertScaleAbs(img, alpha, plus_alpha), cv2.convertScaleAbs(img, alpha,
                                                                                                        minus_alpha)
        cv2.imwrite(join(save_light, basename(image)[:-4] + "_light.png"), plus_contras)
        cv2.imwrite(join(save_dark, basename(image)[:-4] + "_dark.png"), minus_contrast)


def affine_image(path, save_path):
    list_file, _ = get_all_file(path)
    for img in list_file:
        if img.endswith('png'):
            im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            h, w = im.shape[:2]
            if h < w:
                pts1, pts2 = np.float32([[0, 0], [w, 0], [w / 2, h]]), np.float32([[0, h / 2], [w, h / 2], [w / 2, h]])
                M = cv2.getAffineTransform(pts1, pts2)
                img_affine = cv2.warpAffine(im, M, (w, h))
                save = join(save_path, str(basename(img)[:-4]) + '_affine.png')
                crop = img_affine[0 + int(h / 2):0 + h, 0:0 + w]
                cv2.imwrite(save, crop)
            else:
                pts1, pts2 = np.float32([[0, h / 2], [w, 0], [w, h]]), np.float32(
                    [[0, h / 2], [w / 2, 0], [w / 2, h / 2]])
                M = cv2.getAffineTransform(pts1, pts2)
                img_affine = cv2.warpAffine(im, M, (w, h))
                save = join(save_path, str(basename(img)[:-4]) + '_affine.png')
                crop = img_affine[0:0 + h, 0:0 + int(w / 2)]
                cv2.imwrite(save, crop)


def add_plastic_wrap_to_foreground(path_wrap, path_foreground, annotation_num, save_path, rotate=False):
    wraps, _ = get_all_file(path_wrap)  # Get list wrap plastic
    foregrounds, _ = get_all_file(path_foreground)  # Get list foregrounds
    dictionary = {}  # ex: {"1": [1,0]}  => [1,0] : 1 annotation value, 0 rotate degree value
    overlap_value = 3  # Overlap value
    idx = 0  # Image index
    name = "no_name"
    while True:
        if len(foregrounds) == 0:
            break
        bg_path = random.choice(foregrounds)
        print(bg_path)
        id_ = class_id(basename(bg_path))
        bg = Image.open(bg_path).convert("RGBA")
        merged_image = bg.copy()
        bw, bh = merged_image.size  # Get weight height of merge image
        merged_image = merged_image.resize((bw // 2, bh // 2))
        bw, bh = merged_image.size  # Get weight height of merge image
        wrap = random.choice(wraps)
        wrap_img = Image.open(wrap).convert("RGBA")
        wrap_w, wrap_h = wrap_img.size
        ratio = 1
        if bw < wrap_w and bh < wrap_h:
            ratio = bw / wrap_w
        if bw > wrap_w and bh > wrap_h:
            ratio = wrap_w / bw
        wrap_img = wrap_img.resize((int(wrap_w * ratio), int(wrap_h * ratio)))
        print(f"{wrap_img.size} =>> {bg.size}")
        print(wrap)
        if id_ not in dictionary:
            dictionary[id_] = [1, 0]
        else:
            if dictionary[id_][0] > annotation_num:
                print(f"{id_} finnish")
                foregrounds.remove(bg_path)

        merged_image.paste(wrap_img, (0, 0), wrap_img)
        if "light" in basename(bg_path):
            name = join(save_path, f"{basename(bg_path)[:-4]}_light_2_{idx}")
        elif "dark" in basename(bg_path):
            name = join(save_path, f"{basename(bg_path)[:-4]}_dark_2_{idx}")
        is_write = True
        dictionary[id_][0] += 1

        if is_write:
            merged_image.save(name + ".png", format="png")
        idx += 1


def create_dataset(dataset_path, images_path, bg_dir):
    # Init Dataset Directory
    path_img_train, path_img_valid = join(dataset_path, 'images', 'train'), join(dataset_path, 'images', 'valid')
    path_lbl_train, path_lbl_valid = join(dataset_path, 'labels', 'train'), join(dataset_path, 'labels', 'valid')
    os.makedirs(path_img_train, exist_ok=True)
    os.makedirs(path_img_valid, exist_ok=True)
    os.makedirs(path_lbl_train, exist_ok=True)
    os.makedirs(path_lbl_valid, exist_ok=True)

    images, labels = [], []
    for file in os.listdir(images_path):
        images.append(file) if file.endswith('png') else labels.append(file)

    num_of_valid = len(images) // 5
    print(f'Len of images : {len(images)}\nLen of labels : {len(labels)}\nLen valid : {num_of_valid}')

    random_images = []
    for i in range(0, num_of_valid, 1):
        while True:
            img_ran = random.choice(images)
            if img_ran not in random_images:
                random_images.append(img_ran[:-4])
                break

    for img in os.listdir(images_path):
        # Add valid image and label to Dataset Directory
        if img[:-4] in random_images:
            shutil.copy(join(images_path, img), join(path_img_valid, img)) if img.endswith(
                'png') else shutil.copy(join(images_path, img), join(path_lbl_valid, img))

        # Add train image and label to Dataset Directory
        else:
            shutil.copy(join(images_path, img), join(path_img_train, img)) if img.endswith(
                'png') else shutil.copy(join(images_path, img), join(path_lbl_train, img))

    # Add background image to train dataset
    for img in os.listdir(bg_dir):
        shutil.copy(join(bg_dir, img), join(path_img_train, img))


def check_annotation(path_check_labels, path_check_train, path_save):
    files, _ = get_all_file(path_check_labels)
    index = 0

    for f in files:
        if basename(f) == "classes.txt":
            continue
        elif f.endswith("txt"):
            if basename(f)[:-4] + ".png" in os.listdir(path_check_train):
                image = cv2.imread(join(path_check_train, basename(f)[:-4] + ".png"))
                with open(f, "r", encoding="UTF-8") as txt:
                    for line in txt:
                        texts = line.split(" ")
                        save_f = join(path_save, texts[0])
                        os.makedirs(save_f, exist_ok=True)
                        box = (float(texts[1]), float(texts[2]), float(texts[3]), float(texts[4]))
                        x, y, w, h = yolo_box_to_rec_box(box, image.shape[:2])
                        crop = image[y:y + h, x:x + w]
                        index += 1
                        save_name = basename(f)[:-4] + f"_{index}.png"
                        cv2.imwrite(join(save_f, save_name), crop)
            elif basename(f)[:-4] + ".jpg" in os.listdir(path_check_train):
                image = cv2.imread(join(path_check_train, basename(f)[:-4] + ".jpg"))
                with open(f, "r", encoding="UTF-8") as txt:
                    for line in txt:
                        texts = line.split(" ")
                        save_f = join(path_save, texts[0])
                        os.makedirs(save_f, exist_ok=True)
                        box = (float(texts[1]), float(texts[2]), float(texts[3]), float(texts[4]))
                        x, y, w, h = yolo_box_to_rec_box(box, image.shape[:2])
                        crop = image[y:y + h, x:x + w]
                        index += 1
                        save_name = basename(f)[:-4] + f"_{index}.png"
                        cv2.imwrite(join(save_f, save_name), crop)
            else:
                continue


class MergeThread(threading.Thread):
    def __init__(self, bgs, fgs, merge_lock, dictionary_id, save_path="", file_name="merged", annotation_num=1,
                 rotate=False):
        super(MergeThread, self).__init__()
        self.annotation_num = annotation_num
        self.fgs = fgs
        self.bgs = bgs
        self.rotate = rotate
        self.save_path = save_path
        self.file_name = file_name
        self.merge_lock = merge_lock
        self.dictionary_id = dictionary_id

    def run(self):  # Get list foregrounds
        overlap_value = 10  # Overlap value
        idx = 0  # Image index
        name = "no_name"
        while True:
            if len(self.fgs) == 0:  # Break when len of list foreground == 0
                break
            bg_path = random.choice(self.bgs)  # Random choice one of background in list backgrounds
            bg = Image.open(bg_path).convert("RGBA")  # Read bg by Image PIL with 4 channels
            print(f"Processing {basename(bg_path)}")
            merged_image = bg.copy()
            bw, bh = merged_image.size  # Get weight height of merge image
            cur_h, cur_w, max_h, max_w = 0, 0, 0, 0
            while True:
                if len(self.fgs) == 0:  # Break when len of list foreground == 0
                    break
                fg = random.choice(self.fgs)  # Random choice foreground
                id_ = class_id(basename(fg))  # Get id of product
                self.merge_lock.acquire()
                if str(id_) not in self.dictionary_id.keys():
                    self.dictionary_id[id_] = [0, 0]
                self.merge_lock.release()
                if self.dictionary_id[id_][0] <= self.annotation_num:
                    fore_image = Image.open(fg).convert("RGBA")  # Read foreground image with 4 channels
                    if self.rotate:
                        self.merge_lock.acquire()
                        if self.dictionary_id[id_][1] > 359:
                            self.dictionary_id[id_][1] = 0
                        self.dictionary_id[id_][1] += 20
                        self.merge_lock.release()
                        fore_image = fore_image.rotate(self.dictionary_id[id_][1], expand=True)
                        fore_image = fore_image.crop(w_h_image_rotate(fore_image))
                    fw, fh = fore_image.size  # Foreground size
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
                    yolo_box = bnd_box_to_yolo_box(box, (bh, bw))  # Converse Bounding Box(xywh) to Yolo format(xyxy)
                    cls = int(id_)
                    name = join(self.save_path, f"{self.file_name}_{idx}")
                    with open(name + ".txt", 'a') as f:
                        f.write(f"{cls} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
                    cur_w += fw - overlap_value
                    self.merge_lock.acquire()
                    self.dictionary_id[id_][0] += 1
                    self.merge_lock.release()
                else:
                    self.merge_lock.acquire()
                    self.fgs.remove(fg)
                    self.merge_lock.release()
                    print(
                        f" the amount of class '{id_}' is enough. Removed it from foregrounds list. Continue. . .")
                    continue

            merged_image.save(name + ".png", format="png")
            idx += 1


side_bgs = r"\\192.168.0.241\nam\yakult_project\images_processed\background_1\side_bg"
fgs_, _ = get_all_file(r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\resize")
bgs_, _ = get_all_file(r"\\192.168.0.241\nam\yakult_project\images_processed\background_1\side_bg")
save_ = r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\merged"
threads = []
time_ = datetime.now()
merge_lock_ = threading.Lock()
dictionary = {}
for i in range(0, 5, 1):
    t = MergeThread(fgs=fgs_, bgs=bgs_, merge_lock=merge_lock_, dictionary_id=dictionary, save_path=save_,
                    annotation_num=50,
                    file_name=f"merged_{i}")
    t.start()
    threads.append(t)
for t in threads:
    t.join()

print(f"Total time need :  {datetime.now() - time_}")
count_annotation(save_, "classes.txt")

# Resize
# files_, _ = get_all_file(r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\original")
# save = r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\resize"
#
# for image in files:
#     img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
#     for i in np.arange(0.7, 1.2, 0.1):
#         img_rs = cv2.resize(img,(0,0), fx=i, fy=i)
#         for j in range(1,11,1):
#             cv2.imwrite(join(save, basename(image)[:-4] + f"_rs_{round(i,1)}-{j}.png"), img_rs)
# transform = A.Compose([
#     # A.Blur(blur_limit=5),
#     A.RandomBrightnessContrast(p=0.5)
# ])
# contras_,_= get_all_file(r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\blur")
# for f in contras_:
#     image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
#     image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
#     transformed = transform(image=image)
#     transformed_image = transformed["image"]
#     transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGBA2BGRA)
#     cv2.imwrite(join(r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\brightness",
#                      basename(f)[:-4] + "_brightness.png"), transformed_image)

# image = cv2.imread(
#     r"\\192.168.0.241\nam\yakult_project\images_processed\20221019_check_with_parameter\resize\0_03-1_rs_0.7-1.png",
#     cv2.IMREAD_UNCHANGED)
#
# image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
# transformed = transform(image=image)
# transformed_image = transformed["image"]
# transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGBA2BGRA)
# cv2.imwrite("d.png", transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindow()
# fixs, _ = get_all_file(r"\\192.168.0.241\nam\yakult_project\images_processed\20221021\fail_image\side")
# for f in fixs:
#     dir_, base_ = os.path.split(f)
#     new_name = base_.replace("top", "side")
#     old = f
#     new = join(dir_, new_name)
#
#     os.rename(old, new)
