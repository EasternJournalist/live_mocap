import bpy
import pickle
import numpy as np
import tqdm
from mathutils import Matrix
import math

INPUT_FILE = 'tmp\\bone_animation_data.pkl'

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)
fov = data['fov']
frame_rate = data['frame_rate']
bone_names = data['bone_names']
bone_euler_sequence = data['bone_euler_sequence']
location_sequence = data['location_sequence']
scale = data['scale']

all_bone_names = data['all_bone_names']

root = 'pelvis'

bpy.data.objects['Camera'].location = (0, 0, 0)
bpy.data.objects['Camera'].rotation_euler = (math.pi / 2., 0, 0)
bpy.data.objects['Camera'].data.angle = fov

# set frame rate
bpy.context.scene.render.fps = int(frame_rate)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(bone_euler_sequence)

skeleton_objs = list(filter(lambda o: o.type == 'ARMATURE', bpy.data.objects))
assert len(skeleton_objs) == 1, "There should be only one skeleton object"
skeleton = skeleton_objs[0]
skeleton.location = (0, 0, 0)
skeleton.rotation_euler = (-math.pi / 2, 0, 0)
skeleton.scale = (scale, scale, scale)

# apply animation to skeleton
for i in range(len(bone_euler_sequence)):
    for j, b in enumerate(bone_names):
        bone = skeleton.pose.bones[b]
        bone.rotation_mode = 'YXZ'
        bone.rotation_euler = bone_euler_sequence[i][j].tolist()
        bone.keyframe_insert(data_path='rotation_euler', frame=i)
    
    # global location
    x, y, z = location_sequence[i].tolist()
    skeleton.location = x, z, -y
    skeleton.keyframe_insert(data_path='location', frame=i)

# # add keypoints animation
# # create a sphere for each keypoint
# bpy.ops.mesh.primitive_uv_sphere_add(radius=0.005, location=(0, 0, 0))
# sphere = bpy.context.object
# for j, k in enumerate(keypoints_names):
#     anchor = bpy.data.objects.new(k, sphere.data)
#     bpy.context.scene.collection.objects.link(anchor)
#     for i, (kpts, visib) in enumerate(keypoints):
#         if visib[j]:
#             anchor.location = kpts[j, :].tolist()
#             anchor.keyframe_insert(data_path='location', frame=i)