#色を判別する
#115 = midas20_1.jpgのR値の中央値
#最も赤い値を０Mとする

#画像読み込み

#色の判定
#抽出と比較
import json
import cv2
import numpy as np
import cut
import run_copy

#基準２５５を０M、１１５を１０Mとして色を距離に変換する
def distance(x):
    return (255 - x) / 14.0

def output_meters(img_pass, img_json=None, id=None):
    img_pass = run_copy.midas(img_pass, 'test_img/midas')
    if id is None:
        count = cut.cut(img_pass, img_json)
        meters=[]
    else :
        count = cut.cut(img_pass, img_json, id)
    print(count, '枚カットされました')
    for c in range(1, count+1):
        bgr_array = cv2.imread('test_img/' + str(c) + '.jpg')
        r_array = [] #R値だけを入れる
        for m in bgr_array:
            for i in m:
                r_array.append(i[2])

        median = np.median(r_array)
        meter = distance(median)
        print(str(c) + '番の画像は' + str(meter) + 'メートルです。R値の中央値:' + str(median))

        if meter > 10:
            meter = None
            
        if id is None:
            meters.append(meter)
    if id is None:
        count = cut.cut(img_pass, img_json)
        return meters
    else :
        count = cut.cut(img_pass, img_json, id)
        return meter
    

with open('forward_car/json/1.json') as f:
    img_json = json.load(f)
print(output_meters('forward_car/images/1.jpg', img_json, "tnpIj9B7NQ"))
print(output_meters('forward_car/images/1.jpg', img_json))