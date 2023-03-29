'''
This script creates 24 different renderings of a given object.
We have written a new function to determine the camera location.
For illumination, we adhere the settings determined by the TextureFields authors. 
We generate and save not only the RGB and depth, but also the internal&external camera parameters.
The aximuth and elevation angles are also saved
We set image size to 224. We use fixed distance of 1.5 between camera center and origin. (Note that object is centered.)
'''


import argparse
import sys
import os
from math import radians
import bpy
from mathutils import Vector, Color
import colorsys
import numpy as np
import math

# Functions needed for rendering
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def vec_to_blender(vec):
    v1, v2, v3 = np.split(vec, 3, axis=-1)
    vec = np.concatenate([v1, -v3, v2], axis=-1)
    return vec


# This function assigns a camera location for the scene rendering.
# We use sphere mode to render condition images and depths and general mode to request poses.
def get_random_camera(mode, view = 0):
    if mode == 'general':

        cam_r = np.random.uniform(0.7, 1.5)
        cam_loc = np.zeros(3)
        while np.linalg.norm(cam_loc) <= 1e-2:
            cam_loc = np.random.randn(3)
        cam_loc[2] = abs(cam_loc[2])
        cam_loc = cam_loc * cam_r / np.linalg.norm(cam_loc)
        
    elif mode == 'sphere':
        cam_r = 1.5
        az = np.random.normal(view*15,1)
        ele = np.random.normal(25, 5)
        
        az_rad = math.radians(az)
        ele_rad = math.radians(ele)
        
        x = cam_r * math.cos(ele_rad) * math.sin(az_rad)
        y = cam_r * math.cos(ele_rad) * math.cos(az_rad)
        z = cam_r * math.sin(ele_rad)
        
        cam_loc = np.array([x, y, z])
        
        return cam_loc, np.array([az, ele, cam_r])
        
    else:
        raise ValueError('Invalid camera sampling mode "%s"' % mode)

    return cam_loc



def rendering_function(object_name, model_path, save_path):
    
    views = 24
    transform = None
    remove_doubles = True
    edge_split = True
    depth_scale = 0.5
    color_depth = '16'
    format_name = 'OPEN_EXR'
    camera_type = 'sphere'
    
    object_path = model_path + object_name + '/model.obj'
    print(object_path)
    
    # Create output folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set up rendering of depth map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    
    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    bpy.context.scene.render.image_settings.file_format = format_name
    bpy.context.scene.render.image_settings.color_depth = color_depth
    
    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if format_name == 'OPEN_EXR':
        links.new(render_layers.outputs['Z'], depth_file_output.inputs[0])
    else:
        print('Please export depth as exr')
    # Delete default cube
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()
    
    # Load object
    bpy.ops.import_scene.obj(filepath=object_path)
    
    model = bpy.data.objects.new('Model', None)
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp']:
            continue
        object.parent = model
    bpy.context.scene.objects.link(model)
    
    # Load transform
    if transform is not None:
        t0_dict = np.load(transform)
        t0_scale = t0_dict['scale']
        t0_loc = t0_dict['loc']
        bb0_min, bb0_max = t0_dict['bb0_min'], t0_dict['bb0_max']
        bb1_min, bb1_max = t0_dict['bb1_min'], t0_dict['bb1_max']
    else:
        t0_scale = np.ones(3)
        t0_loc = np.zeros(3)
    
    # Modifiers
    for object in model.children:
        bpy.context.scene.objects.active = object
    
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
        if remove_doubles:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        if edge_split:
            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")
    
    # Make light just directional, disable shadows.
    lamp = bpy.data.lamps['Lamp']
    lamp.type = 'HEMI'
    lamp.shadow_method = 'NOSHADOW'
    
    # Possibly disable specular shading:
    lamp.use_specular = False
    lamp.energy = 0.5
    
    
    
    # Rendering options
    scene = bpy.context.scene
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    # scene.render.alpha_mode = 'TRANSPARENT'
    bpy.data.worlds["World"].horizon_color = (1., 1., 1.)
    
    # Set up camera
    cam = scene.objects['Camera']
    cam.location = (0, 1.3, 0.8)
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
    
    # Lamp constraint
    lamp = scene.objects['Lamp']
    lamp_constraint = lamp.constraints.new(type='TRACK_TO')
    lamp_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    lamp_constraint.up_axis = 'UP_Y'
    b_empty2 = parent_obj_to_camera(lamp)
    lamp_constraint.target = b_empty2
    
    # Some output options
    fp = os.path.join(save_path, object_name)
    fp_depth = fp + '/depth_224/'
    fp_image = fp + '/image_224/'
    
    
    if not os.path.exists(fp_depth):
        os.makedirs(fp_depth)
    if not os.path.exists(fp_image):
        os.makedirs(fp_image)
    
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png
    
    stepsize = 360.0 / views
    rotation_mode = 'XYZ'
    
    # albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # albedo_file_output.label = 'Albedo Output'
    
    # links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])
    
    for output_node in [depth_file_output]:
        output_node.base_path = ''
    
    # for output_node in [albedo_file_output]:
    #     output_node.base_path = ''
    
    cameras_world = []
    cameras_projection = []
    blender_T = np.array([
        [1., 0., 0, 0.],
        [0., 0., -1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.]
    ])
    
    
    K0 = 137./2 * np.array([
        [1., 0., 0., 1.],
        [0, 1., 0., 1.],
        [0., 0., 0., 1.],
    ])
    assert(camera_type in ('fixed', 'circle', 'sphere', 'general'))
    
    out_dict = {}
    rendering_parameters = []

    # For each view, we sample render RGB and depth. 
    for i in range(0, views):
        
        # Sample camera location using sphere mode.
        cam_loc, angles = get_random_camera('sphere', i)
        cam.location = Vector((cam_loc[0], cam_loc[1], cam_loc[2]))
        rendering_parameters.append(angles)
        print(cam.location)
    
        # Sample illumination
        lamp_loc = 2 * get_random_camera('general')
        lamp.location = Vector((lamp_loc[0], lamp_loc[1], lamp_loc[2]))
        lamp.location = Vector((0, 0, 10))
        bpy.data.lamps['Lamp'].energy = 1.0 
        
        # Set output filepaths
        scene.render.filepath = os.path.join(fp_image, '{0:03d}'.format(i))
        depth_file_output.file_slots[0].path = os.path.join(fp_depth, '{0:03d}'.format(i))
    
        # Render images
        bpy.ops.render.render(write_still=True)  # render still
    
        # Save camera properties
        cam_M = np.asarray(cam.matrix_world.inverted())
    
        # Blender coordinate convention
        cam_M = cam_M  @ blender_T
        cameras_world.append(cam_M)
    
        cam_P = np.asarray(cam.calc_matrix_camera(
            bpy.context.scene.render.resolution_x,
            bpy.context.scene.render.resolution_y,
            bpy.context.scene.render.pixel_aspect_x,
            bpy.context.scene.render.pixel_aspect_y,
        ))
        
 
        cam_P = K0 @ cam_P
        out_dict['camera_mat_%d' % i] = cam_P
        out_dict['world_mat_%d' % i] = cam_M[:3, :]
    
    # Save camera parameters and angles.
    np.savez(os.path.join(fp_depth, 'cameras.npz'), **out_dict)
    np.save(os.path.join(fp_depth, 'rendering_parameters.npy'), np.array(rendering_parameters))






# Call the blender script
if __name__ == '__main__':
    import sys
    object_name = sys.argv[4]
    model_path = sys.argv[5]
    save_path = sys.argv[6]
    print('Listing')
    print(object_name)
    print(model_path)
    print(save_path)

    rendering_function(object_name, model_path, save_path)
    


