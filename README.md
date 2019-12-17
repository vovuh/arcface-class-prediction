# Image class prediction using pytorch arcface

It's just a reminder to possibly run this project after years.

Firsrly, create a subdirectory *dataset* in the main project directory.
There will be your initial images with corresponding classes.
All images of the same class **must** be in the same subdirectory.
I.e. if you have the class called *class* then its subdirecroty will be *dataset/class*
and all images belonging to it must be in this subdirectory.

Also create a subdirectory *checkpoints* in the main project directory.
There will be all intermediate and final trained models.

Then just run *train.py*, wait 50 epochs (or less, if loss on validate increases).

Choose the best model you have (models are named like this: *checkpoints/general_v7_bottleneck_epoch_loss.h5*).
Choose the best model by *loss* (this is loss on validate), change the variable *model_path* in [this line](https://github.com/vovuh/arcface-class-prediction/blob/9638cc5422180699a09e0adb5dabc8567c98483b/evaluate.py#L47)
and then just run *evaluate.py*, after this run *visualize.py*.

Unfortunately, now you should download all needed dependencies by yourself. Maybe sometime I will fix this...
