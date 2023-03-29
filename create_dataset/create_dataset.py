import argparse
import sys
import os
from math import radians
import colorsys
import numpy as np
import math
from subprocess import call


# The following paths should be adjusted before using the code.


# Model path. ShapeNetCorev1 is used for getting 3d shapes and textures. 
# 02958343 is the folder name for car dataset.
model_path = '/srv/beegfs02/scratch/texture/data/ShapeNetCore.v1/02958343/'

# Output path, the results will be saved in the following folder.
save_path = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/3rdparty/texture_fields/shapenet/synthetic_cars_nospecular/'

# Name of the blender script. 
rendering_script = '/srv/beegfs02/scratch/texture/data/single_view_3d_w_texture/src/create_dataset/render_blender.py'


# List the objects in the directory.
objects = os.listdir(save_path)
# We get rid of lst files in the directory
objects.remove('test.lst')
objects.remove('train.lst')
objects.remove('val.lst')
objects.sort()

print('Listing all objects')
print(objects)
print('All objects listed')


# For each object in the list, call the blender script separately using subprocess.
for object_name in objects:
    cmd = 'blender --background --python %s %s %s %s' % (rendering_script, object_name, model_path, save_path)
    call(cmd, shell=True)
