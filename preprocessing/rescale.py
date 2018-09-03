import argparse
import os
import subprocess
import time
import sys
import ipdb
import pickle
from utils.meter import *


def main(args):
    # Parameters from the args
    dir, h, w, fps, suffix = args.dir, args.height, args.width, args.fps, args.suffix

    # Video dir
    dir_split = dir.split('/')
    video_dir = dir_split[-1]
    root_dir = '/'.join(dir_split[:-1])
    new_video_dir = "{}_{}x{}_{}".format(video_dir, w, h, fps)
    new_dir = os.path.join(root_dir, new_video_dir)
    os.makedirs(new_dir, exist_ok=True)

    # load the existing dict if exist
    dict_video_length_fn = os.path.join(new_dir, 'dict_id_length.pickle')
    if os.path.isfile(dict_video_length_fn):
        with open(dict_video_length_fn, 'rb') as file:
            dict_video_length = pickle.load(file)
    else:
        dict_video_length = {}

    # Get the initial video filenames
    list_video_fn = get_all_videos(dir, suffix)
    print("\n### Initial directory: {} ###".format(dir))
    print("=> {} videos in total\n".format(len(list_video_fn)))

    # Loop over the super_video and extract
    op_time = AverageMeter()
    start = time.time()
    list_error_fn = []
    for i, video_fn in enumerate(list_video_fn):
        try:
            # Rescale
            rescale_video(video_fn, w, h, fps, dir, new_dir, suffix, dict_video_length, ffmpeg=args.ffmpeg,
                          crf=args.crf)

            # Log
            duration = time.time() - start
            op_time.update(duration, 1)
            time_done = get_time_to_print(op_time.avg * (i + 1))
            time_remaining = get_time_to_print(op_time.avg * len(list_video_fn))
            print('[{0}/{1}] : Time {batch_time.val:.3f} ({batch_time.avg:.3f}) [{done} => {remaining}]\t'.format(
                    i + 1, len(list_video_fn), batch_time=op_time,
                    done=time_done, remaining=time_remaining))
            sys.stdout.flush()
            start = time.time()
        except:
            print("Impossible to rescale_videos super_video for {}".format(video_fn))
            list_error_fn.append(video_fn)

    print("\nDone!")
    print("\nImpossible to rescale {} videos: \n {}".format(len(list_error_fn), list_error_fn))

    # Save the dict id -> length
    with open(dict_video_length_fn, 'wb') as file:
        pickle.dump(dict_video_length, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("\nLength of each video stored ---> {}".format(dict_video_length_fn))

    # Print
    print("\n### You can now have access to your videos rescaled => {} ###\n".format(new_dir))


def get_duration(file):
    """Get the duration of a super_video using ffprobe. -> https://stackoverflow.com/questions/31024968/using-ffmpeg-to-obtain-super_video-durations-in-python"""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(file)
    output = subprocess.check_output(
        cmd,
        shell=True,  # Let this run in the shell
        stderr=subprocess.STDOUT
    )
    return float(output)


def rescale_video(video_fn, w, h, fps, dir, new_dir, suffix, dict_video_length, ffmpeg, crf=17):
    """ Rescale a video according to its new width, height an fps """

    # Output video_name
    video_id = video_fn.replace(dir, '').replace(suffix, '')
    video_fn_rescaled = video_fn.replace(dir, new_dir)
    video_fn_rescaled = video_fn_rescaled.replace(suffix, suffix.lower())

    # Create the dir
    video_dir_to_create = '/'.join(video_fn_rescaled.split('/')[:-1])
    os.makedirs(video_dir_to_create, exist_ok=True)

    # Check if the file already exists
    if os.path.isfile(video_fn_rescaled):
        print("{} already exists".format(video_fn_rescaled))
    else:
        subprocess.call(
            '{ffmpeg} -i {video_input} -vf scale={w}:{h} -crf {crf} -r {fps} -y {video_output} -loglevel panic'.format(
                ffmpeg=ffmpeg,
                video_input=video_fn,
                h=h,
                w=w,
                fps=fps,
                video_output=video_fn_rescaled,
                crf=crf
            ), shell=True)

        # Get the duration of the new super_video (in sec)
        duration_sec = get_duration(video_fn_rescaled)
        duration_frames = int(duration_sec * fps)

        # update the dict id -> length
        dict_video_length[video_id] = duration_frames

    return video_fn_rescaled


def get_all_videos(dir, extension='mp4'):
    """ Return a list of the videos filename from a directory and its subdirectories """

    list_video_fn = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(extension)]:
            # Make sure it is not a hidden file
            if filename[0] != '.':
                fn = os.path.join(dirpath, filename)
                list_video_fn.append(fn)

    return list_video_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing')
    parser.add_argument('--dir', metavar='DIR',
                        default='../data/vlog/videos',
                        help='Path to the videos dir')
    parser.add_argument('--width', default=256, type=int,
                        metavar='W', help='Width of  of the output videos')
    parser.add_argument('--height', default=256, type=int,
                        metavar='H', help='Height of the output videos')
    parser.add_argument('--fps', default=30, type=int,
                        metavar='FPS',
                        help='Frames per second of the output video')
    parser.add_argument('--suffix', metavar='E',
                        default='.mp4',
                        help='Suffix of all the videos files - default version for the VLOG dataset')
    parser.add_argument('--crf', default=17, type=int,
                        metavar='CRF',
                        help='Quality of the compressing - lower is better (default: 17)')
    parser.add_argument('--ffmpeg', metavar='FF',
                        default='ffmpeg',
                        help='Path to your ffmpeg to use (default: ffmpeg)')

    args = parser.parse_args()

    main(args)
