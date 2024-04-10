import numpy as np
from transnetv2 import TransNetV2
from pathlib import Path
import sys
from tqdm import tqdm


model = TransNetV2()

video_dir = '/data/lijw/data/video_data/UCF-101/videos'
save_root = "results/UCF-101/shots"
Path(save_root).mkdir(parents=True, exist_ok=True)

path_list = list(Path(video_dir).glob('**/*.*'))
print(f"共有视频{len(path_list)}个")

for file in tqdm(path_list):
    pred_save_path = Path(save_root).joinpath(file.name + '.predictions.txt')
    scene_save_path = Path(save_root).joinpath(file.name + '.scenes.txt')
    file = str(file)
    if pred_save_path.exists() or scene_save_path.exists():
        print(f"[TransNetV2] {pred_save_path} or {scene_save_path} already exists. "
                f"Skipping video {file}.", file=sys.stderr)
        continue

    video_frames, single_frame_predictions, all_frame_predictions = \
        model.predict_video(file)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    np.savetxt(pred_save_path, predictions, fmt="%.6f")

    scenes = model.predictions_to_scenes(single_frame_predictions)
    np.savetxt(scene_save_path, scenes, fmt="%d")
