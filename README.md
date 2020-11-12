# Image class prediction using pytorch arcface

It's just a reminder to possibly run this project after years.

First, create a subdirectory *dataset* in the main project directory.
There will be your initial images with corresponding classes.
All images of the same class **must** be in the same subdirectory.
I.e. if you have the class called *class* then its subdirectory will be *dataset/class*
and all images belonging to it must be in this subdirectory.

Also create a subdirectory *checkpoints* in the main project directory.
There will be all intermediate and final trained models.

Then just run *train.py*, wait 50 epochs (or less, if loss on validate increases).

**DON'T FORGET TO CHANGE THE NUMBER OF CLASSES IN THE CONFIG**.

Choose the best model you have (models are named like this: *checkpoints/general_v7_bottleneck_epoch_loss.h5*).
Choose the best model by *loss* (this is loss on validate), change the variable *model_path* in the last line of *evaluate.py*
and then just run it. After that, run *visualize.py* to see results.

Unfortunately, now you should download all needed dependencies by yourself. Maybe I will fix this some day...
