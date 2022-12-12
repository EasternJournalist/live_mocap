import bpy
import pickle
import numpy as np
import tqdm
from mathutils import Matrix
import math

INPUT_FILE = 'tmp\\bone_animation_data.pkl'

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)
scales = data['scales']
keypoints_names = data['keypoints_names']
keypoints = data['keypoints']
optim_bone_names = data['optim_bone_names']
optim_bone_euler = data['optim_bone_eulers']
bone_names = data['all_bone_names']
bone_matrix_world = data['all_bone_matrix_world']

root = 'pelvis'

bpy.data.objects['Camera'].location = (0, 0, 0)
bpy.data.objects['Camera'].rotation_euler = (math.pi, 0, 0)

# set frame rate
bpy.context.scene.render.fps = 30
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = optim_bone_euler.shape[0]

skeleton_objs = list(filter(lambda o: o.type == 'ARMATURE', bpy.data.objects))
assert len(skeleton_objs) == 1, "There should be only one skeleton object"
skeleton = skeleton_objs[0]
skeleton.matrix_world = Matrix.Identity(4)

# apply animation to skeleton
for i in tqdm.trange(optim_bone_euler.shape[0]):
    for j, bone_name in enumerate(optim_bone_names):
        bone = skeleton.pose.bones[bone_name]
        bone.rotation_mode = 'YXZ'
        bone.rotation_euler = optim_bone_euler[i, j, :].tolist()
        bone.keyframe_insert(data_path='rotation_euler', frame=i)
    
    # global location and scale
    skeleton.pose.bones[root].matrix = bone_matrix_world[i, bone_names.index(root)].T.tolist()
    skeleton.pose.bones[root].keyframe_insert(data_path='location', frame=i)
    skeleton.pose.bones[root].scale = [scales[i]] * 3
    skeleton.pose.bones[root].keyframe_insert(data_path='scale', frame=i)

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