#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import glob
from moviepy.editor import *
import cv2

#画像を動画にする
def write_video(video_file, video_path=r'./input_video/'):

    #元動画のFPSを取得する
    cap = cv2.VideoCapture(video_path + video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(video_fps)
    duretion = 1 / cap.get(cv2.CAP_PROP_FPS)
    print(duretion)

    # inputディレクトリ以下の拡張子が.jpgのファイル名リストを一括取得
    file_list = glob.glob(r'./output/*.png')
    # ファイル名リストを昇順にソート
    file_list.sort()

    # スライドショーを作る元となる静止画情報を格納する処理
    clips = [] 
    for m in file_list:
        clip = ImageClip(m).set_duration(duretion)
        clips.append(clip)

    # スライドショーの動画像を作成する処理
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(r"./output_video/output.mp4", 
                                fps=video_fps,
                                write_logfile=True,
                                )


#動画を画像にする
def video_2_frames(video_file, video_path=r'./input_video/', image_dir='./input/', image_file='img_%s.png'):
    # Video to frames
    i = 0
    cap = cv2.VideoCapture(video_path + video_file)
    if cap.isOpened():
        print("true")
    else :
        print("false")
    while(cap.isOpened()):
        flag, frame = cap.read()  # Capture frame-by-frame
        if flag == False:  # Is a frame left?
            break
        cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)  # Save a frame
        print('Save', image_dir+image_file % str(i).zfill(6))
        i += 1

    cap.release()  # When everything done, release the capture

