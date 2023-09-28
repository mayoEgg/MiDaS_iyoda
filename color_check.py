#色を判別する
#115 = midas20_1.jpgのR値の中央値
#最も赤い値を０Mとする

#画像読み込み

#色の判定
#抽出と比較
import cv2
import numpy as np

def distance(x):
    return (255 - x) / 14.0

bgr_array = cv2.imread('cropped_images/midas19_1.jpg')
r_array = [] #R値だけを入れる
for m in bgr_array:
    for i in m:
        r_array.append(i[2])

median = np.median(r_array)
m = distance(median)

print('この画像は' + str(m) + 'メートルです。R値の中央値:' + str(median))