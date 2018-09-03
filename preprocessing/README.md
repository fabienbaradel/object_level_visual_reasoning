# Preprocessing the videos

First step is to rescale the videos.
We will rescale the entire datasets to a resolution of 256x256 and a fps of 30.
Such operation can be one using `ffmpeg`.

## VLOG dataset
In the following I will assume that you have download the VLOG dataset and the dataset is located in the directory `/data/vlog/videos`.
I have attached 3 videos of the dataset in the Github repo to make sure you can replicate the operation on your own dataset, below is the paths to the two videos:
* `/data/vlog/videos/O/B/o/v_eAalQCMBOBo/049/clip.mp4`
* `/data/vlog/videosa/q/g/v_4jqvo3TQaqg/016/clip.mp4`
* `/data/vlog/videosa/q/g/v_4jqvo3TQaqg/016/clip.mp4`

For applying the preprocessing on the full directory and create a new one you can run the following command:
```
cd preprocessing
python rescale.py --width 256 --height 256 --fps 30
```

It will create a new directory named `/data/vlog/videos_256x256_30` with the same structure of the initial one.
You will also get a pickle file called `dict_id_length.pickle` in your new directory, the length of each videos will be saved into this dictionnary (it will be used while training/testing). 
Feel free to check the `rescale.py` to check other possible arguments (e.g. crf, ffmpeg version, suffix).
The generic command is:
```
python rescale.py \
--dir <LOC-VIDEO-DIR> \
--width <OUTPUT-W> \
--height <OUTPUT-H> \
--fps <OUTPUT-FPS> \
--crf <OUTPUT-QUALITY> \
--ffmpeg <YOUR-FFMPEG> \
--suffix <SUFFIX-OF-ALL-VIDEOS>
```

You can now open the original video and the rescaled one using the following command:
```
# Original one
open /data/vlog/videos/O/B/o/v_eAalQCMBOBo/049/clip.mp4
du -sh /data/vlog/videos/O/B/o/v_eAalQCMBOBo/049/clip.mp4

# Rescaled one
open /data/vlog/videos_256x256_30/O/B/o/v_eAalQCMBOBo/049/clip.mp4
du -sh /data/vlog/videos_256x256_30/O/B/o/v_eAalQCMBOBo/049/clip.mp4
```

This preprocessing step allows us to reduce the size of our video by a certain amount as you can see by running the above lines (`du -sh ...`).

On the real VLOG dataset it takes ~15 hours to rescale the entire dataset.

You are now ready to test the input pipeline -> [README_DATALOADER.md](../loader/README.md)