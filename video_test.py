import glob
from moviepy.editor import *
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture('./input_video/airport.mp4')
    
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