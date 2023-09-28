#ライブラリの読み込み
import os
import cv2
import json
import glob
import re

def create_dict(filepaths):
    file_dict = {}
    for filepath in filepaths:
        if 'pfm' not in filepath:
            file_dict[re.sub("-.*", "", os.path.splitext(os.path.basename(filepath))[0])] = filepath
    
    return file_dict

def croppe_image(original_width, original_height, value_x, value_y, value_width, value_height, img, file_name, number):
    x = int(original_width * value_x / 100)
    y = int(original_height * value_y / 100)
    width = int(original_width * value_width / 100)
    height = int(original_height * value_height / 100)
    #画像切り抜き
    cropped_image = img[y:y+height, x:x+width]
    #画像保存
    cv2.imwrite('cropped_images/' + str(file_name) + '_' + str(number) + '.jpg', cropped_image)

#forward_car/imagesディレクトリのすべての画像の読み込み
image_filepaths = create_dict(glob.glob(r'./forward_car/images/*'))
json_filepaths = create_dict(glob.glob(r'./forward_car/json/*'))
midas_filepaths = create_dict(glob.glob(r'./output/*'))

for file_name in image_filepaths.keys() & json_filepaths.keys():
    #画像読み込み
    original_img = cv2.imread(image_filepaths[file_name])
    midas_img = cv2.imread(midas_filepaths[file_name])
    #jsonファイルの読み込み
    with open(json_filepaths[file_name]) as f:
        json_data = json.load(f)

    number = 1
    #jsonファイルを参照して画像を切り抜く
    for item in json_data['annotations'][0]['result']:
        value = item['value']
        if value['rectanglelabels'][0] == '前方車':
            croppe_image(item['original_width'], item['original_height'], value['x'], value['y'], value['width'], value['height'], original_img, file_name, number)
            croppe_image(midas_img.shape[1], midas_img.shape[0], value['x'], value['y'], value['width'], value['height'], midas_img, 'midas' + file_name, number)
            number += 1