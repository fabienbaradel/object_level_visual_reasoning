# Testing - Training

## VLOG

### Training from scratch
Feel free to train the model by yourself using the following script:

```shell
# Path to the VLOG dataset
VLOG=$VLOG # update with you own path

# Training - generic command
./training_vlog.sh <MY-RESUME>

# Training - my command
my_resume=/home/fbaradel/log_eccv18
./training_vlog.sh $my_resume

```
where `<MY-RESUME>` is the path to your resume.
First you will train the object head (10 epochs) and then you will train the full model (10 epochs).

### Testing
We release weights of our model pretrained on VLOG: [link](https://drive.google.com/open?id=12rSb41HGKm_u93isZJArG-RukLpPmJFI).
Move the checkpoint to a resume directory.
```shell
# Pythonpath
PYTHONPATH=.

# Generic command
python main.py --root <LOC-VLOG-DATA> --resume <PATH-YOUR-RESUME> \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--depth 50 \
-t 4 \
-b 16 \
--cuda \
--dataset vlog \
--heads object+context \
-j 4 \
-e 

# My command
python main.py --root $VLOG --resume /home/fbaradel/logdir/eccv18_rebuttal/vlog/two_heads/object_coco_50 \
--blocks 2D_2D_2D_2.5D \
--object-head 2D \
--add-background \
--train-set train+val \
--arch orn_two_heads \
--depth 50 \
-t 4 \
-b 16 \
--cuda \
--dataset vlog \
--heads object+context \
-j 4 \
-e 
```
where `<PATH-YOUR-RESUME>` is the location of the directory where the pretrained model is located.
