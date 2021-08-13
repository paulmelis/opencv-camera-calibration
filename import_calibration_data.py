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

scene.render.resolution_x = j['image_resolution'][0]
scene.render.resolution_y = j['image_resolution'][1]

# Chessboard

chessboard_points = numpy.array(j['chessboard_points'], 'float32')

chessboard_mesh = bpy.data.meshes.new(name='chessboard corners')
chessboard_mesh.vertices.add(chessboard_points.shape[0])
chessboard_mesh.vertices.foreach_set('co', chessboard_points.flatten())
chessboard_mesh.update()
#if chessboard_mesh.validate(verbose=True):
#    print('Mesh data did not validate!')
    
chessboard_object = bpy.data.objects.new(name='chessboard corners', object_data=chessboard_mesh)
bpy.context.scene.collection.objects.link(chessboard_object)    

# Cameras

camera_collection = bpy.data.collections['Collection']
camera_collection.name = 'Cameras'

if 'sensor_size_mm' not in j:
    print("Warning: camera sensor size value not available, set it manually!")

for img_file, values in j['chessboard_orientations'].items():
    
    camdata = bpy.data.cameras.new(name=img_file)            
    if 'sensor_size_mm' in j:
        camdata.sensor_width = j['sensor_size_mm'][0]
    camdata.lens_unit = 'FOV'
    camdata.angle = radians(j['fov_degrees'][0])

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
    
