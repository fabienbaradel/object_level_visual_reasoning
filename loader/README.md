# Iterating over your dataset efficiently

Let's now test our dataset loader.
I assume that you have already finished step 1 (rescaling the videos).

You can now run the following line:
```
cd loader
python test.py --dataset vlog

# Generic command
python test.py \
--dataset <DATASET-NAME> \ # vlog or epic
--root <ROOT-DIR-DATASET> \ # on the above example it is ../data/vlog
--t <VIDEO-LENGTH>
```
If you have a new PNG file called `img.png` in this directory it means that you are able to iterate over your videos!
The first time you run the `test.py` file it should take some time because it is looping over the directory to make sure that all the videos are present and retrieve their label and length.

ps: the code for visualization of the masks has been adapted from [detectorch](https://github.com/ignacio-rocco/detectorch/blob/master/lib/utils/vis.py)

You can start training and evaluating on this dataset => [README_training](../README_TRAINING_TESTING.md)