# 抽取关键帧，保存关键帧图片
import base64
import numpy as np
from pathlib import Path
import cv2
import json
from io import BytesIO
from PIL import Image
import traceback
from tqdm import tqdm

def PIL_to_base64(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, 'JPEG', quality=100)
        base64_data = base64.b64encode(buffer.getvalue()).decode()
    return base64_data

def base64_to_file(base64_data: str, save_path: str):
    bytes_data = base64.b64decode(base64_data)
    with open(save_path, 'wb') as f:
        f.write(bytes_data)


class keyframeExtract:
    """
    关键帧提取类
    """
    def __init__(self, video_path, shot_index_list, preds):
        self.video_path = video_path
        self.frame_time = [] # 关键帧时间
        self.frame_base64 = [] # 关键帧
        self.frame_indexes = [] # 关键帧标号
        self.shot_time = [] # 镜头分割时间
        self.shot_base64 = [] # 镜头分割帧
        self.shot_indexes = shot_index_list # 镜头分割帧标号
        self.shot_period_indexes = [] # 镜头分割帧标号
        self.preds = preds # 预测概率

    def get_keyframe_index_by_preds(self, preds, start_index, end_index):
        """根据累加预测概率，动态自适应选取关键帧"""
        ACCUMULATE_THRESHOLD = 0.5
        curr_preds = preds[start_index:end_index+1]
        # 当连续累加概率大于ACCUMULATE_THRESHOLD时，分割为子镜头
        subshot_index_list = [0]
        sum_preds = 0
        for i in range(len(curr_preds)):
            sum_preds += curr_preds[i]
            if sum_preds >= ACCUMULATE_THRESHOLD:
                subshot_index_list.append(i)
                sum_preds = 0
        subshot_index_list.append(len(curr_preds) - 1)
        # 选取每个子镜头的中间帧作为关键帧
        keyframe_index_list = []
        for i in range(len(subshot_index_list) - 1):
            keyframe_index_list.append((subshot_index_list[i] + subshot_index_list[i + 1]) // 2)
        # 每个帧标号都加上初始标号
        keyframe_index_list = [i + start_index for i in keyframe_index_list]
        # 过滤掉重复的帧标号，并保持顺序
        keyframe_index_list = list(dict.fromkeys(keyframe_index_list))
        return keyframe_index_list

    def get_keyframe_index_by_interval(self, keyframe_step, start_index, end_index):
        """每隔固定帧数选取一个关键帧"""
        shot_frame_cnt = end_index - start_index + 1
        keyframe_num = (shot_frame_cnt + keyframe_step // 5) // keyframe_step
        keyframe_index_list = []
        if keyframe_num == 0:
            frame_index = int((start_index + end_index) / 2)
            keyframe_index_list.append(frame_index)
        else:
            for i in range(keyframe_num):
                frame_index = start_index + keyframe_step // 2 + keyframe_step * i
                keyframe_index_list.append(frame_index)
        return keyframe_index_list

    def parse_keyframe(self, keyframe_step=None):
        cap = cv2.VideoCapture(str(self.video_path))
        self.total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总共的帧数
        # assert self.total_frame == self.shot_indexes[-1] + 1, f"total frame count {self.total_frame} is not equal to the last shot index {self.shot_indexes[-1] + 1}"
        if self.total_frame != self.shot_indexes[-1] + 1:
            # print(f"WARNING: total frame count {self.total_frame} is not equal to the last shot index {self.shot_indexes[-1] + 1}")
            self.shot_indexes[-1] = self.total_frame - 1
        pre_shot_end_index = -1
        self.shot_time.append(0)
        self.shot_period_indexes.append(0)
        for i in self.shot_indexes: # 这个shot_indexes没有第一帧，有最后一帧标号
            try:
                curr_shot_end_index = i
                curr_shot_start_index = pre_shot_end_index + 1
                if curr_shot_start_index > curr_shot_end_index:
                    print(f"WARNING: shot start index {curr_shot_start_index} is greater than shot end index {curr_shot_end_index}")
                    continue
                self.shot_period_indexes.append(curr_shot_end_index)
                self.shot_period_indexes.append(curr_shot_end_index + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_shot_end_index)
                success, shot_frame1 = cap.read()
                # RGB_img1 = Image.fromarray(cv2.cvtColor(shot_frame1, cv2.COLOR_BGR2RGB))
                self.shot_time.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
                # success, shot_frame2 = cap.read()
                # RGB_img2 = Image.fromarray(cv2.cvtColor(shot_frame2, cv2.COLOR_BGR2RGB))
                # self.shot_base64.append([PIL_to_base64(RGB_img1), PIL_to_base64(RGB_img2)])

                # 获取关键帧标号
                if keyframe_step is not None:
                    frame_index_list = self.get_keyframe_index_by_interval(keyframe_step, curr_shot_start_index, curr_shot_end_index)
                else:
                    frame_index_list = self.get_keyframe_index_by_preds(self.preds, curr_shot_start_index, curr_shot_end_index)
                # 获取关键帧
                frame_base64_temp = []
                frame_time_temp = []
                frame_index_temp = []
                for frame_index in frame_index_list:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, key_frame = cap.read()
                    if success:
                        RGB_img = Image.fromarray(cv2.cvtColor(key_frame, cv2.COLOR_BGR2RGB))
                        frame_base64_temp.append(PIL_to_base64(RGB_img))
                        frame_time_temp.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
                        frame_index_temp.append(frame_index)
                pre_shot_end_index = curr_shot_end_index
                self.frame_indexes.append(frame_index_temp)
                self.frame_base64.append(frame_base64_temp)
                self.frame_time.append(frame_time_temp)
            except Exception as e:
                print(f"Video path: {self.video_path}, shot index: {i}, error: {e}")
                traceback.print_exc()

        self.shot_period_indexes = self.shot_period_indexes[:-1]
        cap.release()


def shots2keyframes(shots_path, keyframe_save_path):
    with open(shots_path, 'r') as f:
        shot_info = json.load(f)
    data = {}
    for item in tqdm(shot_info):
        filename = item['filename']
        video_id = item['videoId']
        for shot in item['shots']:
            start_time = shot['startTime']
            end_time = shot['endTime']
            start_index = shot['startIndex']
            end_index = shot['endIndex']
            for key in shot['keyFrames']:
                frame_id = key['frameId']
                frame_time = key['frameTime']
                frame_index = key['frameIndex']
                data[frame_id] = {
                    'videoId': video_id,
                    'filename': filename,
                    'startTime': start_time,
                    'endTime': end_time,
                    'startIndex': start_index,
                    'endIndex': end_index,
                    'frameTime': frame_time,
                    'frameIndex': frame_index
                }
    with open(keyframe_save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def iterate_videos(scene_list, video_root, save_keyframe_dir, shots_save_path, keyframe_step):
    data = []
    for v_id, path in tqdm(enumerate(scene_list, 1), total=len(scene_list)):
        video_path = Path(video_root).joinpath('.'.join(path.name.split('.')[:-2]))
        pred_path = Path(path).with_name(Path(path).name.replace('.scenes.txt', '.predictions.txt'))
        shots = np.loadtxt(path, dtype=np.int32, ndmin=2)
        preds = np.loadtxt(pred_path, dtype=np.float32, ndmin=2)
        v = keyframeExtract(str(video_path), shots[:,1].tolist(), preds[:,0].tolist())
        v.parse_keyframe(keyframe_step)

        info = {}
        info['filename'] = video_path.name
        info['videoId'] = v_id
        info['frameCount'] = v.total_frame
        info['shotCount'] = len(v.frame_base64)
        info['shots'] = []
        key_id = 1
        for i_shot in range(len(v.frame_base64)):
            shot_info = {}
            shot_info['startTime'] = v.shot_time[i_shot]
            shot_info['endTime'] = v.shot_time[i_shot + 1]
            shot_info['startIndex'] = v.shot_period_indexes[i_shot * 2]
            shot_info['endIndex'] = v.shot_period_indexes[i_shot * 2 + 1]
            shot_info['keyFrames'] = []
            for i_key in range(len(v.frame_base64[i_shot])):
                frame_id = int(str(v_id) + '%05d'%(key_id))
                key_id += 1
                shot_info['keyFrames'].append(
                    {
                        'frameId': frame_id,
                        'frameTime': v.frame_time[i_shot][i_key],
                        'frameIndex': v.frame_indexes[i_shot][i_key]
                    }
                )
                base64_to_file(v.frame_base64[i_shot][i_key], save_keyframe_dir.joinpath(f'{frame_id}.jpg').as_posix())
            info['shots'].append(shot_info)
        data.append(info)
    with open(shots_save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


data_name = 'BBC_Planet_Earth'
video_root = '/data/lijw/data/video_data/BBC_Planet_Earth_Dataset/videos/'
predict_root = "results/BBC_Planet_Earth/shots"
pred_list = list(Path(predict_root).glob('*.predictions.txt'))
scene_list = list(Path(predict_root).glob('*.scenes.txt'))

suffix = '100'
keyframe_step = 100 # 每多少帧选一个关键帧。当设为非常大的数时，即为选取中间帧为关键帧
save_dir = Path('.').joinpath('results').joinpath(data_name)
save_dir.mkdir(parents=True, exist_ok=True)
save_keyframe_dir = save_dir.joinpath(f'keyframe_{suffix}') # 保存keyframe文件夹路径
save_keyframe_dir.mkdir(parents=True, exist_ok=True)
shots_save_path = save_dir.joinpath(f'shots_{suffix}.json')
keyframe_save_path = save_dir.joinpath(f'keyframe_info_{suffix}.json')

iterate_videos(scene_list, video_root, save_keyframe_dir, shots_save_path, keyframe_step)
shots2keyframes(shots_save_path, keyframe_save_path)
