import os
import glob
import tqdm
import argparse
import numpy as np
from pathlib import Path

import metrics_utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate TransNet")
    # parser.add_argument("log_dir", help="path to log dir")
    # parser.add_argument("epoch", help="what weights to use", type=int)
    parser.add_argument("directory", help="path to the test dataset")
    parser.add_argument("--thr", default=0.5, type=float, help="threshold for transition")
    args = parser.parse_args()

    print(args)
    # gin.parse_config_file(os.path.join(args.log_dir, "config.gin"))
    # options = training.get_options_dict(create_dir_and_summaries=False)

    # if options["original_transnet"]:
    #     net = models.OriginalTransNet()
    #     logit_fc = lambda x: tf.nn.softmax(x)[:, :, 1]
        
    # else:
    #     net = transnet.TransNetV2()
    #     logit_fc = tf.sigmoid

    # @tf.function(autograph=False)
    # def predict(batch):
    #     one_hot = net(tf.cast(batch, tf.float32)[tf.newaxis])
    #     if isinstance(one_hot, tuple):
    #         one_hot = one_hot[0]
    #     return logit_fc(one_hot)[0]

    # net(tf.zeros([1] + options["input_shape"], tf.float32))
    # net.load_weights(os.path.join(args.log_dir, "weights-{:d}.h5".format(args.epoch)))
    # files = glob.glob(os.path.join(args.directory, "*.npy"))

    results = []
    total_stats = {"tp": 0, "fp": 0, "fn": 0}

    # dataset_name = [i for i in args.directory.split("/") if i != ""][-1]
    # img_dir = os.path.join(args.log_dir, "results", "{}-epoch{:d}".format(dataset_name, args.epoch))
    # os.makedirs(img_dir, exist_ok=True)

    files = Path(args.directory).glob('*.scenes.txt')
    for np_fn in tqdm.tqdm(files):
        np_fn = str(np_fn)
        # predictions = []
        ev_scenes = np.loadtxt(np_fn, dtype=np.int32)
        # predictions = data[:, 0]
        # frames = np.load(np_fn)

        # for batch in get_batches(frames):
        #     one_hot = predict(batch)
        #     predictions.append(one_hot[25:75])

        # predictions = np.concatenate(predictions, 0)[:len(frames)]
        #TODO
        # import pdb
        # pdb.set_trace()
        # gt_fn = glob.glob(os.path.join('/data/lijw/data/video_data/BBC_Planet_Earth_Dataset/annotations/shots/', Path(np_fn).stem.split('.')[0] + "*"))[0]
        # gt_fn = glob.glob(os.path.join('/data/lijw/code/TransNetV2/training/consolidated/BBCDataset/', Path(np_fn).stem.split('.')[0] + "*.txt"))[0]
        # gt_fn = glob.glob(os.path.join('/data/lijw/code/TransNetV2/training/consolidated/RAIDataset/', Path(np_fn).stem.split('.')[0] + "*.txt"))[0]
        gt_fn = glob.glob(os.path.join('/data/lijw/code/TransNetV2/training/consolidated/ClipShotsTest/', Path(np_fn).stem.split('.')[0] + "*.txt"))[0]
        # print('NAME:', gt_fn)
        gt_scenes = np.loadtxt(gt_fn, dtype=np.int32, ndmin=2)

        _, _, _, (tp, fp, fn), fp_mistakes, fn_mistakes = metrics_utils.evaluate_scenes(
            gt_scenes, ev_scenes,
            return_mistakes=True)

        total_stats["tp"] += tp
        total_stats["fp"] += fp
        total_stats["fn"] += fn

        # if len(fp_mistakes) > 0 or len(fn_mistakes) > 0:
        #     print(f'ERROR!!!!!!!!!!!!!!  {np_fn}')
            # print(f'ERROR!!!!!!!!!!!!!! fp_mistakes: {fp_mistakes}, fn_mistakes: {fn_mistakes}')
            # img = visualization_utils.visualize_errors(
            #     frames, predictions,
            #     create_dataset.scenes2zero_one_representation(gt_scenes, len(frames))[1],
            #     fp_mistakes, fn_mistakes)
            # if img is not None:
            #     img.save(os.path.join(img_dir, os.path.basename(np_fn[:-3]) + "png"))

        # results.append((np_fn, predictions, gt_scenes))

    # with open(os.path.join(args.log_dir, "results", "{}-epoch{:d}.pickle".format(dataset_name, args.epoch)), "wb") as f:
    #     pickle.dump(results, f)

    p = total_stats["tp"] / (total_stats["tp"] + total_stats["fp"])
    r = total_stats["tp"] / (total_stats["tp"] + total_stats["fn"])
    f1 = (p * r * 2) / (p + r)
    print(f"""
    Precision:{p*100:5.2f}%
    Recall:   {r*100:5.2f}%
    F1 Score: {f1*100:5.2f}%
    """)
