import sys, json, os
from math import radians
import numpy
import bpy
from mathutils import Matrix

scene = bpy.context.scene

# Clear scene!
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

try:
    idx = sys.argv.index('--')
except ValueError:
    print('usage: blend -P %s -- calib.json' % sys.argv[0])
    sys.exit(-1)
    
if idx == len(sys.argv):
    print('usage: blend -P %s -- calib.json' % sys.argv[0])
    sys.exit(-1)
    
# Load calibration file

calib_file = sys.argv[idx+1]
j = json.load(open(calib_file, 'rt'))

calib_dir = os.path.split(calib_file)[0]

# Scene resolution

W, H = j['image_resolution']

scene.render.resolution_x = W
scene.render.resolution_y = H

# Chessboard

# Corner vertices
chessboard_points = numpy.array(j['chessboard_points'], 'float32')

chessboard_mesh = bpy.data.meshes.new(name='chessboard corners')
chessboard_mesh.vertices.add(chessboard_points.shape[0])
chessboard_mesh.vertices.foreach_set('co', chessboard_points.flatten())
chessboard_mesh.update()
#if chessboard_mesh.validate(verbose=True):
#    print('Mesh data did not validate!')
    
chessboard_object = bpy.data.objects.new(name='chessboard corners', object_data=chessboard_mesh)
bpy.context.scene.collection.objects.link(chessboard_object)    

# Textured quad

spacing = j['chessboard_spacing_m']
corners = j['chessboard_inner_corners']

vertices = numpy.array([
    -spacing,           -spacing,               0,
    spacing*corners[0], -spacing,               0,
    spacing*corners[0], spacing*corners[1],     0,
    -spacing,           spacing*corners[1],     0
], 'float32')
indices = numpy.array([0, 1, 2, 3], 'uint32')
loop_start = numpy.array([0], 'uint32')
loop_total = numpy.array([4], 'uint32')
uvs = numpy.array([
    0, 0, 
    1, 0,
    1, 1, 
    0, 1
], 'float32')

m = bpy.data.meshes.new(name='chessboard')
m.vertices.add(4)
m.vertices.foreach_set('co', vertices)
m.loops.add(4)
m.loops.foreach_set('vertex_index', indices)
m.polygons.add(1)
m.polygons.foreach_set('loop_start', loop_start)
m.polygons.foreach_set('loop_total', loop_total)
uv_layer = m.uv_layers.new(name='uvs')
uv_layer.data.foreach_set('uv', uvs)
m.update()
if m.validate(verbose=True):
    print('Mesh data did not validate!')
    
mat = bpy.data.materials.new('chessboard')
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
texcoord = nodes.new(type='ShaderNodeTexCoord')
texcoord.location = 0, 300
mapping = nodes.new(type='ShaderNodeMapping')
mapping.location = 200, 300
mapping.inputs['Scale'].default_value = (corners[0]+1, corners[1]+1, 1)
checktex = nodes.new(type='ShaderNodeTexChecker')
checktex.location = 400, 300
checktex.inputs['Color2'].default_value = 0, 0, 0, 1
checktex.inputs['Scale'].default_value = 1.0
emission = nodes.new(type='ShaderNodeEmission')
emission.location = 600, 300
node_output = nodes.new(type='ShaderNodeOutputMaterial')
node_output.location = 800, 300
links = mat.node_tree.links
links.new(texcoord.outputs['UV'], mapping.inputs['Vector'])
links.new(mapping.outputs['Vector'], checktex.inputs['Vector'])
links.new(checktex.outputs['Color'], emission.inputs['Color'])
links.new(emission.outputs['Emission'], node_output.inputs['Surface'])

m.materials.append(mat)

o = bpy.data.objects.new(name='chessboard', object_data=m)
bpy.context.scene.collection.objects.link(o)    
# Hide by default
o.hide_set(True)

# Cameras

camera_collection = bpy.data.collections['Collection']
camera_collection.name = 'Cameras'

if 'sensor_size_mm' not in j:
    print('Warning: camera sensor size value not available, you need to set it manually!')
if 'fov_degrees' not in j:
    print('Warning: camera FOV not available, you need to set it manually!')

for img_file, values in j['chessboard_orientations'].items():
    
    camdata = bpy.data.cameras.new(name=img_file)            
    
    if 'sensor_size_mm' in j:
        camdata.sensor_fit = 'HORIZONTAL'
        camdata.sensor_width = j['sensor_size_mm'][0]
                
    if 'fov_degrees' in j:
        camdata.lens_unit = 'FOV'
        camdata.angle = radians(j['fov_degrees'][0])    
        
    M = j['camera_matrix']
    fx = M[0][0]
    fy = M[1][1]
    cx = M[0][2]
    cy = M[1][2]
    
    pixel_aspect = fy / fx
    if pixel_aspect > 1:
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = pixel_aspect
    else:
        scene.render.pixel_aspect_x = 1.0 / pixel_aspect
        scene.render.pixel_aspect_y = 1.0 
        
    # Thanks to https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
    camdata.shift_x = -(cx / W - 0.5)
    camdata.shift_y = (cy - 0.5 * H) / W
    
    camobj = bpy.data.objects.new(img_file, camdata)
    
    t = values['translation']
    R = Matrix(values['rotation_matrix'])
    
    # Object xform (chessboard into camera space)
    #camobj.matrix_world = Matrix.Translation(t) @ R.to_4x4()

    # Camera xform (inverse, camera into chessboard space)
    camobj.matrix_world = (Matrix.Rotation(radians(180), 4, 'X') @ Matrix.Translation(t) @ R.to_4x4()).inverted()
    
    # Background image
    basename, ext = os.path.splitext(img_file)
    img_file_original = os.path.join(calib_dir, img_file)
    if os.path.isfile(img_file_original):
        img = bpy.data.images.load(img_file_original)
        assert img is not None
        camdata.show_background_images = True
        bg = camdata.background_images.new()
        bg.image = img
    else:
        print("Image %s not available, can't set as background on camera" % img_file)
    
    camera_collection.objects.link(camobj)
    
