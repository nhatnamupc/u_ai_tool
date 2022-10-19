dic = [str(cls) for cls in range(0, 21, 1)]
print(dic)
import os.path
from utils import general
import cv2
import shutil

join = os.path.join
ROOT = r"\\192.168.0.241\nam\yakult_project\Dataset\dataset_20220921"

# labels = join(ROOT, "side", "labels")
# txt, _ = general.get_all_file(labels)
# #
# old_cls_side = [-99, -1, -2, -3, 3, 4, 25, 26, 28, 29, 31, 32, 34, 21, 23, 7, 10, 13, 16, 19, 73, 74, 75, 76, 77, 78,
#                 79,
#                 80, 81, 82, 83, 84, 85, 86, 87, 88, -4, 1, 42, 43, 44, 20, 22, 40, 41, 45, 33, -5, -6, -7, -8, -9, -10,
#                 -11, -12, -13, -14, -15, -16, 47]
#
old_cls_top = [0, 42, 43, 44, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 22, 40, 41, 45, 33, 24, 25, 27, 28, 30, 31,
               35, 36, 37, 38, 39, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
               67, 68, 69, 70, 71, 72, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15]
#
print(len(old_cls_top))
change_cls = {}
for index, cls in enumerate(old_cls_top):
    change_cls[str(cls)] = index
#
# txts = []
# pngs = []


# for file in txt:
#     if file.endswith("txt"):
#         with open(file, "r", encoding="utf-8") as f:
#             data = [line.strip() for line in f]
#         with open(file, "w+") as f1:
#             for text in data:
#                 texts = text.split(" ")
#                 if int(texts[0]) == 46:
#                     # txts.append(file)
#                     # pngs.append(file[:-4])
#                     new_line = f"{59} {texts[1]} {texts[2]} {texts[3]} {texts[4]}"
#                     f1.writelines(new_line)
#                     f1.writelines("\n")
#                 elif str(texts[0]) in change_cls.keys():
#                     new_line = f"{change_cls[str(texts[0])]} {texts[1]} {texts[2]} {texts[3]} {texts[4]}"
#                     f1.writelines(new_line)
#                     f1.writelines("\n")

# files, _ = general.get_all_file(r"\\192.168.0.241\nam\yakult_project\20221005_data\top")
# save = r"\\192.168.0.241\nam\yakult_project\20221005_data\square\square_images"
# for f in files:
#     print(f)
# if f.endswith("png"):
#     img = cv2.imread(f)
#     h, w = img.shape[:2]
#     pad = abs(w - h)
#     if w > h:
#         padded_img = cv2.copyMakeBorder(
#             img, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[128, 128, 128])
#     else:
#         padded_img = cv2.copyMakeBorder(
#             img, 0, 0, 0, pad, borderType=cv2.BORDER_CONSTANT, value=[128, 128, 128])
#     cv2.imwrite(join(save, os.path.basename(f)[:-4] + "_square.png"), padded_img)
#     old, new = f, join(r"\\192.168.0.241\nam\yakult_project\20221005_data\square", "images", os.path.basename(f))
#     shutil.copy(old, new)
