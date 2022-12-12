import os
import sys
import ctypes
import json
import time
from typing import Dict, List, Tuple
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from utils3d import euler_angles_to_matrix, mls_smooth

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from skeleton_config import load_skeleton_data, get_optimization_target, get_constraints, get_align_location, get_align_scale, MEDIAPIPE_KEYPOINTS_WITH_HANDS, MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS


@torch.jit.script
def barrier(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    return torch.exp(4 * (x - b)) + torch.exp(4 * (a - x))

def eval_matrix_world(parents: torch.Tensor, matrix_bones: torch.Tensor, matrix_basis: torch.Tensor) -> torch.Tensor:
    "Deprecated"
    matrix_bones, matrix_basis = matrix_bones.unbind(), matrix_basis.unbind()
    matrix_world = []
    for i in range(len(matrix_bones)):
        local_mat = torch.mm(matrix_bones[i], matrix_basis[i])
        m = local_mat if parents[i] < 0 else torch.mm(matrix_world[parents[i]], local_mat)
        matrix_world.append(m)
    return torch.stack(matrix_world)

class EvalMatrixWorld(torch.autograd.Function):
    """
    Call c++ function to evaluate matrix_world, for speed. Second order derivative is not supported.
    """

    cdll = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'cpp_eval_bone_matrix/cpp_eval_bone_matrix.dll'))
    cpp_eval_matrix_world = cdll.eval_matrix_world
    cpp_grad_matrix_world = cdll.grad_matrix_world

    @staticmethod
    def forward(ctx, parents: torch.Tensor, matrix_bones: torch.Tensor, matrix_basis: torch.Tensor):
        assert parents.dtype == torch.int64 and parents.is_contiguous()
        assert matrix_bones.dtype == torch.float32 and matrix_bones.is_contiguous()
        assert matrix_basis.dtype == torch.float32 and matrix_basis.is_contiguous()

        matrix_world = torch.zeros_like(matrix_bones)

        EvalMatrixWorld.cpp_eval_matrix_world(
            ctypes.c_ulonglong(parents.shape[0]),
            ctypes.c_void_p(parents.data_ptr()),
            ctypes.c_void_p(matrix_bones.data_ptr()),
            ctypes.c_void_p(matrix_basis.data_ptr()),
            ctypes.c_void_p(matrix_world.data_ptr()),
        )
        ctx.save_for_backward(parents, matrix_bones, matrix_basis, matrix_world)
        return matrix_world

    @staticmethod
    def backward(ctx, grad_out):
        assert grad_out.dtype == torch.float32 and grad_out.is_contiguous()

        parents, matrix_bones, matrix_basis, matrix_world = ctx.saved_tensors
        grad_matrix_basis = torch.zeros_like(matrix_basis)
        grad_matrix_world = grad_out.clone()
        EvalMatrixWorld.cpp_grad_matrix_world(
            ctypes.c_ulonglong(parents.shape[0]),
            ctypes.c_void_p(parents.data_ptr()),
            ctypes.c_void_p(matrix_bones.data_ptr()),
            ctypes.c_void_p(matrix_basis.data_ptr()),
            ctypes.c_void_p(matrix_world.data_ptr()),
            ctypes.c_void_p(grad_matrix_basis.data_ptr()),
            ctypes.c_void_p(grad_matrix_world.data_ptr()),
        )
        return None, None, grad_matrix_basis

eval_matrix_world = EvalMatrixWorld.apply


class SkeletonIKSovler:
    def __init__(self, model_path: str, track_hands: bool = True, **kwargs):
        # load skeleton model data
        all_bone_names, all_bone_parents, all_bone_matrix_world_rest, all_bone_matrix, skeleton_remap = load_skeleton_data(model_path)
        
        self.keypoints = MEDIAPIPE_KEYPOINTS_WITH_HANDS if track_hands else MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS

        # skeleton structure info
        self.all_bone_names: List[str] = all_bone_names
        self.all_bone_parents: List[str] = all_bone_parents
        self.all_bone_parents_id = torch.tensor([(all_bone_names.index(all_bone_parents[b]) if all_bone_parents[b] is not None else -1) for b in all_bone_parents], dtype=torch.long)
        self.all_bone_matrix: torch.Tensor = torch.from_numpy(all_bone_matrix).float()
  
        # Optimization target
        bone_subset, optimizable_bones, kpt_pairs_id, joint_pairs_id = get_optimization_target(all_bone_parents, skeleton_remap, track_hands)
        self.joint_pairs_a, self.joint_pairs_b = joint_pairs_id[:, 0], joint_pairs_id[:, 1]
        self.kpt_pairs_a, self.kpt_pairs_b = kpt_pairs_id[:, 0], kpt_pairs_id[:, 1]
        self.bone_parents_id = torch.tensor([(bone_subset.index(all_bone_parents[b]) if all_bone_parents[b] is not None else -1) for b in bone_subset], dtype=torch.long)
        subset_id = [all_bone_names.index(b) for b in bone_subset]
        self.bone_matrix = self.all_bone_matrix[subset_id]

        # joint constraints
        joint_constraint_id, joint_constraint_value = get_constraints(all_bone_names, all_bone_matrix_world_rest, optimizable_bones, skeleton_remap)
        self.joint_contraint_id = joint_constraint_id
        self.joint_constraints_min, self.joint_constraints_max = joint_constraint_value[:, :, 0], joint_constraint_value[:, :, 1]

        # align location
        self.align_location_kpts, self.align_location_bones = get_align_location(all_bone_names, skeleton_remap)

        # align scale
        self.align_scale_pairs_kpt, self.align_scale_pairs_bone = get_align_scale(all_bone_names, skeleton_remap)
        rest_joints = torch.from_numpy(all_bone_matrix_world_rest)[:, :3, 3]
        self.align_scale_pairs_length = torch.norm(rest_joints[self.align_scale_pairs_bone[:, 0]] - rest_joints[self.align_scale_pairs_bone[:, 1]], dim=-1)
        
        # optimization hyperparameters
        self.lr = kwargs.get('lr', 1.0)
        self.max_iter = kwargs.get('max_iter', 24)
        self.tolerance_change = kwargs.get('tolerance_change', 1e-6)
        self.tolerance_grad = kwargs.get('tolerance_grad', 1e-4)
        self.joint_constraint_loss_weight = kwargs.get('joint_constraint_loss_weight', 0.1)
        self.pose_reg_loss_weight = kwargs.get('pose_reg_loss_weight', 0.01)
        self.smooth_steps = kwargs.get('smooth_steps', 12)

        # optimizable bone euler angles
        self.optimizable_bones = optimizable_bones
        self.gather_id = torch.tensor([(optimizable_bones.index(b) + 1 if b in optimizable_bones else 0) for b in bone_subset], dtype=torch.long)[:, None, None].repeat(1, 4, 4)
        self.all_gather_id = torch.tensor([(optimizable_bones.index(b) + 1 if b in optimizable_bones else 0) for b in all_bone_names], dtype=torch.long)[:, None, None].repeat(1, 4, 4)
        self.optim_bone_euler = torch.zeros((len(optimizable_bones), 3), requires_grad=True)

        # smoothness
        self.joint_trace = []
        # self.kpts_trace = []
        self.align_location = torch.zeros(3)
        self.align_scale = torch.tensor(0.0)

    def fit(self, kpts: torch.Tensor, valid: torch.Tensor):
        # smooth keypoints
        # self.kpts_trace.append(kpts)
        # if len(self.kpts_trace) > self.smooth_steps:
        #     self.kpts_trace.pop(0)
        # kpts_sm = self.kpts_trace[-1] if len(self.kpts_trace) <= 3 else mls_smooth(self.kpts_trace)
        self.align_location = kpts[self.align_location_kpts].mean(dim=0)

        optimizer = torch.optim.LBFGS(
            [self.optim_bone_euler], 
            line_search_fn='strong_wolfe', 
            lr=self.lr, 
            max_iter=100 if len(self.joint_trace) == 0 else self.max_iter, 
            tolerance_change=self.tolerance_change, 
            tolerance_grad=self.tolerance_grad
        )

        pair_valid = valid[self.kpt_pairs_a] & valid[self.kpt_pairs_b]
        kpt_pairs_a, kpt_pairs_b = self.kpt_pairs_a[pair_valid], self.kpt_pairs_b[pair_valid]
        joint_pairs_a, joint_pairs_b = self.joint_pairs_a[pair_valid], self.joint_pairs_b[pair_valid]

        kpt_dir = kpts[kpt_pairs_a] - kpts[kpt_pairs_b]
        kpt_pairs_length = torch.norm(kpts[self.align_scale_pairs_kpt[:, 0]] - kpts[self.align_scale_pairs_kpt[:, 1]], dim=-1)
        align_scale = (kpt_pairs_length / self.align_scale_pairs_length).mean()
        if align_scale > 0:
            self.align_scale = align_scale
            kpt_dir = kpt_dir / self.align_scale

        def _loss_closure():
            optimizer.zero_grad()
            optim_matrix_basis = euler_angles_to_matrix(self.optim_bone_euler, 'YXZ')
            matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0, index=self.gather_id)
            matrix_world = eval_matrix_world(self.bone_parents_id, self.bone_matrix, matrix_basis)
            joints = matrix_world[:, :3, 3]
            joint_dir = joints[joint_pairs_a] - joints[joint_pairs_b]
            dir_loss = F.mse_loss(kpt_dir, joint_dir)
            joint_prior_loss = barrier(self.optim_bone_euler[self.joint_contraint_id], self.joint_constraints_min, self.joint_constraints_max).mean()
            pose_reg_loss = self.optim_bone_euler.square().mean()
            loss = dir_loss + self.pose_reg_loss_weight * pose_reg_loss + self.joint_constraint_loss_weight * joint_prior_loss 
            loss.backward()
            return loss

        if len(kpt_dir) > 0:
            optimizer.step(_loss_closure)
        self.joint_trace.append(self.optim_bone_euler.detach().clone())
        if len(self.joint_trace) > self.smooth_steps:
            self.joint_trace.pop(0)

    def get_bone_euler(self) -> torch.Tensor:
        joints_smoothed = self.joint_trace[-1] if len(self.joint_trace) <= 3 else mls_smooth(self.joint_trace)
        return joints_smoothed
    
    def get_scale(self) -> float:
        return self.align_scale

    def get_bone_matrix_world(self) -> torch.Tensor:
        optim_matrix_basis = euler_angles_to_matrix(self.get_bone_euler(), 'YXZ')
        matrix_basis = torch.gather(torch.cat([torch.eye(4).unsqueeze(0), optim_matrix_basis]), dim=0, index=self.all_gather_id)
        matrix_world = eval_matrix_world(self.all_bone_parents_id, self.all_bone_matrix, matrix_basis)

        # align scale and location
        matrix_world = torch.tensor([self.align_scale, self.align_scale, self.align_scale, 1.])[None, :, None] * matrix_world
        matrix_world[:, :3, 3] += self.align_location - matrix_world[self.align_location_bones, :3, 3].mean(dim=0)
        return matrix_world


def update_eval_matrix(bone_parents: torch.Tensor, bone_matrix_world: torch.Tensor, updated_bones: Dict[int, torch.Tensor] = None):
    bone_matrix_world_updated = bone_matrix_world.clone()
    for i, matrix in updated_bones.items():
        if matrix.shape == (3, 3):
            bone_matrix_world_updated[i, :3, :3] = matrix
        elif matrix.shape == (4, 4):
            bone_matrix_world_updated[i] = matrix
        else:
            raise ValueError('Invalid matrix shape')
    to_update = set(updated_bones.keys())
    for i in range(bone_matrix_world.shape[0]):
        if bone_parents[i].item() in to_update:
            bone_matrix_world_updated[i] = bone_matrix_world_updated[bone_parents[i]] @ (bone_matrix_world[bone_parents[i]].inverse() @ bone_matrix_world[i])
    return bone_matrix_world_updated


def test():
    import tqdm

    solver = SkeletonIKSovler('D:\\projects\\morphing/avatar/wei/', track_hands=True)
    with open('tmp/kpts3ds_mengnan.pkl', 'rb') as f:
        body_keypoints = pickle.load(f)

    bone_eulers_seq, bone_matrix_world_seq, scale_seq = [], [], []
    start_t = None 
    for kpts3d, valid in tqdm.tqdm(body_keypoints):
        solver.fit(torch.from_numpy(kpts3d).float(), torch.from_numpy(valid).bool())
        bone_matrix_world_seq.append(solver.get_bone_matrix_world())
        bone_eulers_seq.append(solver.get_bone_euler())
        scale_seq.append(solver.get_scale())
        if start_t is None:
            start_t = time.time()
    print(f'time per frame: {(time.time() - start_t) / (len(body_keypoints) - 1)}')

    with open('tmp/bone_animation_data.pkl', 'wb') as f:
        pickle.dump({
            'keypoints_names': solver.keypoints,
            'keypoints': body_keypoints,
            'scales': torch.stack(scale_seq).numpy(),
            'optim_bone_names': solver.optimizable_bones,
            'optim_bone_eulers': torch.stack(bone_eulers_seq).numpy(),
            'all_bone_names': solver.all_bone_names,
            'all_bone_matrix_world': torch.stack(bone_matrix_world_seq).numpy(),
        }, f)

    np.save(
        'tmp/bone_matrice_sequence.npy',
        torch.stack(bone_matrix_world_seq).numpy()
    )


if __name__ == '__main__':
    test()
