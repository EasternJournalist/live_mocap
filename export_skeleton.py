import os
import bpy
import numpy as np
import json

SAVE_DIR = "tmp\\skeleton"
os.makedirs(SAVE_DIR, exist_ok=True)

def iterdfs(bone):
    yield bone
    for child in bone.children:
        for descent in iterdfs(child):
            yield descent

def iterbones(bones):
    for r in filter(lambda b: b.parent is None, bones):
        for b in iterdfs(r):
            yield(b)

def export_bones(skeleton):
    bones = skeleton.pose.bones
    bone_names = [b.name for b in iterbones(bones)]
    bone_parents = {b: bones[b].parent.name if bones[b].parent is not None else None for b in bone_names}

    bone_matrix_rel, bone_matrix_world = [], []
    for bn in bone_names:
        b = bones[bn]
        bone_matrix_world.append(np.array(skeleton.matrix_world @ b.matrix, dtype=np.float32))
        if b.parent is None:
            m = np.array(skeleton.matrix_world @ b.matrix @ b.matrix_basis.inverted(), dtype=np.float32)
        else:
            m = np.array(b.parent.matrix.inverted() @ b.matrix @ b.matrix_basis.inverted(), dtype=np.float32)
        bone_matrix_rel.append(m)
    return bone_names, bone_parents, np.stack(bone_matrix_rel), np.stack(bone_matrix_world)

def export_numpy(info: dict, prefix = []):
    for k in info:
        prefix_ = prefix + [k]
        if isinstance(info[k], dict):
            export_numpy(info[k], prefix=prefix_)
        elif isinstance(info[k], np.ndarray):
            # make sure the array is C contiguous
            print(k, info[k].dtype, info[k].shape, info[k].flags.c_contiguous)
            filename = str('_').join(prefix_) + '.npy'
            np.save(os.path.join(SAVE_DIR, filename), info[k])
            info[k] = filename

def save_json_with_numpy(info: dict, file):
    export_numpy(info)
    with open(file, 'w') as f:
        json.dump(info, f, indent=4)

skeleton_objs = list(filter(lambda o: o.type == 'ARMATURE', bpy.data.objects))
assert len(skeleton_objs) == 1, "There should be only one skeleton object"
bone_names, bone_parents, bone_matrix_rel, bone_matrix_world = export_bones(skeleton_objs[0])

# Save skeleton
skeleton = {
    'bone_names': bone_names,
    'bone_parents': bone_parents,
    'bone_matrix_rel': bone_matrix_rel,
    'bone_matrix_world': bone_matrix_world,
    'bone_remap': {
        'left_ear': None,
        'right_ear': None,
        "left_eye": None,
        "right_eye": None,
        "head": None,
        "neck": None,
        "left_collar": None,
        "right_collar": None,
        "left_shoulder": None,
        "right_shoulder": None,
        "left_elbow": None,
        "right_elbow": None,
        "left_wrist": None,
        "right_wrist": None,
        
        "spine1": None,
        "spine2": None,
        "spine3": None,
        "pelvis": None,

        "left_hip": None,
        "right_hip": None,
        "left_knee": None,
        "right_knee": None,
        "left_ankle": None,
        "right_ankle": None,
        "left_toe": None,
        "right_toe": None,

        "left_index1": None, "left_index2": None, "left_index3": None, 'left_index4': None,
        "left_middle1": None, "left_middle2": None, "left_middle3": None, 'left_middle4': None,
        "left_ring1": None, "left_ring2": None, "left_ring3": None, 'left_ring4': None,
        "left_pinky1": None, "left_pinky2": None, "left_pinky3": None, 'left_pinky4': None,
        "left_thumb1": None, "left_thumb2": None, "left_thumb3": None, 'left_thumb4': None,

        "right_index1": None, "right_index2": None, "right_index3": None, 'right_index4': None,
        "right_middle1": None, "right_middle2": None, "right_middle3": None, 'right_middle4': None,
        "right_ring1": None, "right_ring2": None, "right_ring3": None, 'right_ring4': None,
        "right_pinky1": None, "right_pinky2": None, "right_pinky3": None, 'right_pinky4': None,
        "right_thumb1": None, "right_thumb2": None, "right_thumb3": None, 'right_thumb4': None,
    }
}

save_json_with_numpy(skeleton, os.path.join(SAVE_DIR, 'skeleton.json'))

quit()