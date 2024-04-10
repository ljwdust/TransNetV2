from pathlib import Path
import numpy as np
from tqdm import tqdm
from video_utils import get_frames
from visualization_utils import visualize_scenes


# video_path = '/data/lijw/data/video_data/test/821d154b9f1547e59be54a897043aa8c.mp4'
# scenes_path = '/data/lijw/data/video_data/test/821d154b9f1547e59be54a897043aa8c.mp4.scenes.txt'
# save_path = '/data/lijw/data/video_data/test/821d154b9f1547e59be54a897043aa8c.mp4-transnetv2-scenes.png'

data_root = '/data/lijw/data/video_data/test/TransNetV2_result'
video_root = '/data/lijw/data/video_data/test/videos'
scene_list = Path(data_root).glob('*.scenes.txt')
for scene_path in tqdm(scene_list):
    video_path = Path(video_root).joinpath(scene_path.name.replace('.scenes.txt', ''))
    save_path = scene_path.with_suffix('.png')
    if save_path.exists():
        continue
    scenes = np.loadtxt(scene_path, dtype=np.int32, ndmin=2)
    video = get_frames(video_path)
    visualize_scenes(video, scenes).save(save_path)
