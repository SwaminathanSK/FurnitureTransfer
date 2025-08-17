"""
Dense reward system for one-leg furniture assembly task.
Based on reward engineering principles from:
- "Local Policies Enable Zero-shot Long-horizon Manipulation" (arXiv:2410.22332)
- "Twisting Lids Off with Two Hands" (arXiv:2403.02338)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import furniture_bench.utils.transform as T


def quat_pos_to_transform_tensor(quat, pos):
    """Convert quaternion and position to 4x4 transformation matrix."""
    # Convert quaternion to rotation matrix
    rot_mat = T.quat2mat(quat.cpu().numpy())
    
    # Create 4x4 transformation matrix
    transform = torch.eye(4, device=quat.device)
    transform[:3, :3] = torch.from_numpy(rot_mat).to(quat.device)
    transform[:3, 3] = pos
    
    return transform


def mat_to_quat_tensor(rot_mat):
    """Convert rotation matrix to quaternion tensor."""
    quat = T.mat2quat(rot_mat.cpu().numpy())
    return torch.from_numpy(quat).to(rot_mat.device)


class OneLegDenseReward:
    """Dense reward system for one-leg furniture assembly with multi-stage decomposition."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Reward weights (tunable hyperparameters)
        self.weights = {
            'approach': 2.0,        # Approaching the leg
            'grasp': 5.0,          # Grasping the leg
            'pick': 3.0,           # Lifting the leg
            'transport': 4.0,       # Moving towards target
            'align': 6.0,          # Fine alignment
            'screw': 8.0,          # Screwing motion
            'assemble': 10.0,      # Final assembly
            'efficiency': 1.0,     # Time efficiency bonus
            'safety': 2.0,         # Safety penalties
        }
        
        # Task-specific thresholds
        self.thresholds = {
            'approach_dist': 0.15,      # Distance to start approaching
            'grasp_dist': 0.03,         # Distance to attempt grasp
            'pick_height': 0.02,        # Minimum lift height
            'transport_dist': 0.20,     # Distance to start transport phase
            'align_dist': 0.18,         # Distance for fine alignment (much larger)
            'align_angle': 0.2,         # Angular alignment threshold (radians)
            'screw_dist': 0.16,         # Distance to start screwing (much larger)
            'screw_angle': 0.15,        # Rotational motion for screwing
            'assemble_dist': 0.015,     # Final assembly distance
            'assemble_angle': 0.1,      # Final assembly angle
        }
        
        # Stage tracking
        self.current_stage = 'approach'
        self.stage_completion = {
            'approach': False,
            'grasp': False, 
            'pick': False,
            'transport': False,
            'align': False,
            'screw': False,
            'assemble': False
        }
        
        # Previous values for progress tracking
        self.prev_distances = {}
        self.step_count = 0
        self.prev_leg_quat = None  # For tracking rotational motion during screwing
        
    def reset(self):
        """Reset reward state for new episode."""
        self.current_stage = 'approach'
        self.stage_completion = {stage: False for stage in self.stage_completion}
        self.prev_distances = {}
        self.step_count = 0
        self.prev_leg_quat = None
        
    def compute_dense_reward(self, 
                           gripper_pos: torch.Tensor,
                           gripper_quat: torch.Tensor, 
                           parts_poses: torch.Tensor,
                           gripper_state: torch.Tensor,
                           assembled: bool = False) -> Tuple[float, Dict]:
        """
        Compute dense reward for one-leg assembly task.
        
        Args:
            gripper_pos: (3,) gripper position
            gripper_quat: (4,) gripper quaternion  
            parts_poses: (num_parts * 7,) flattened part poses [pos, quat]
            gripper_state: (2,) gripper finger positions
            assembled: boolean indicating if parts are assembled
            
        Returns:
            total_reward: scalar reward
            reward_info: dict with reward components
        """
        self.step_count += 1
        
        # Extract part positions and orientations
        table_pos = parts_poses[0:3]  # part 0: table top
        table_quat = parts_poses[3:7]
        leg_pos = parts_poses[4*7:4*7+3]  # part 4: leg to assemble  
        leg_quat = parts_poses[4*7+3:4*7+7]
        
        # Target assembly position (relative to table)
        target_pos, target_quat = self._get_assembly_target(table_pos, table_quat)
        
        # Compute reward components
        rewards = {}
        
        # 1. Approach Phase Reward
        rewards['approach'] = self._approach_reward(gripper_pos, leg_pos)
        
        # 2. Grasp Phase Reward  
        rewards['grasp'] = self._grasp_reward(gripper_pos, leg_pos, gripper_state)
        
        # 3. Pick Phase Reward
        rewards['pick'] = self._pick_reward(leg_pos, gripper_state)
        
        # 4. Transport Phase Reward
        rewards['transport'] = self._transport_reward(leg_pos, target_pos, gripper_state)
        
        # 5. Alignment Phase Reward
        rewards['align'] = self._alignment_reward(leg_pos, leg_quat, target_pos, target_quat, gripper_state)
        
        # 6. Screwing Phase Reward
        rewards['screw'] = self._screwing_reward(leg_pos, leg_quat, target_pos, target_quat, gripper_state)
        
        # 7. Assembly Phase Reward
        rewards['assemble'] = self._assembly_reward(leg_pos, leg_quat, target_pos, target_quat, assembled)
        
        # 8. Efficiency Bonus
        rewards['efficiency'] = self._efficiency_bonus()
        
        # 9. Safety Penalties
        rewards['safety'] = self._safety_penalty(gripper_pos, parts_poses)
        
        # Combine weighted rewards
        total_reward = sum(self.weights[key] * reward for key, reward in rewards.items())
        
        # Update stage tracking
        self._update_stage_tracking(gripper_pos, leg_pos, target_pos, gripper_state, assembled)
        
        return total_reward, rewards
    
    def _get_assembly_target(self, table_pos: torch.Tensor, table_quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get target position and orientation for leg assembly."""
        # Based on SquareTable assembled_rel_poses - leg4 goes to corner
        rel_pos = torch.tensor([0.05625, 0.046875, 0.05625], device=self.device)
        rel_quat = torch.tensor([0, 0, 0, 1], device=self.device)  # No rotation
        
        # Transform to world coordinates
        table_mat = quat_pos_to_transform_tensor(table_quat, table_pos)
        rel_mat = quat_pos_to_transform_tensor(rel_quat, rel_pos)
        target_mat = table_mat @ rel_mat
        
        target_pos = target_mat[:3, 3]
        target_quat = mat_to_quat_tensor(target_mat[:3, :3])
        
        return target_pos, target_quat
    
    def _approach_reward(self, gripper_pos: torch.Tensor, leg_pos: torch.Tensor) -> float:
        """Reward for approaching the leg."""
        dist = torch.norm(gripper_pos - leg_pos)
        
        if dist < self.thresholds['grasp_dist']:
            return 1.0  # Reached grasp position
        elif dist < self.thresholds['approach_dist']:
            # Smooth reward based on progress
            progress = 1.0 - (dist - self.thresholds['grasp_dist']) / (self.thresholds['approach_dist'] - self.thresholds['grasp_dist'])
            return 0.5 + 0.5 * progress
        else:
            # Long-range attraction
            return max(0.0, 0.5 - dist.item() / 0.5)
    
    def _grasp_reward(self, gripper_pos: torch.Tensor, leg_pos: torch.Tensor, gripper_state: torch.Tensor) -> float:
        """Reward for grasping the leg."""
        dist = torch.norm(gripper_pos - leg_pos)
        grasp_closed = (gripper_state < 0.03).all()  # Gripper closed
        
        if dist < self.thresholds['grasp_dist']:
            if grasp_closed:
                return 1.0  # Successfully grasped
            else:
                return 0.5  # Close but not grasped
        else:
            return 0.0
    
    def _pick_reward(self, leg_pos: torch.Tensor, gripper_state: torch.Tensor) -> float:
        """Reward for picking up the leg."""
        # Handle both scalar and tensor gripper_state
        if isinstance(gripper_state, torch.Tensor):
            if gripper_state.numel() == 1:
                grasp_closed = gripper_state.item() < 0.03
            else:
                grasp_closed = (gripper_state < 0.03).all()
        else:
            grasp_closed = gripper_state < 0.03
        
        if not grasp_closed:
            return 0.0
            
        # Check if leg is lifted above initial height
        if 'initial_leg_z' not in self.prev_distances:
            self.prev_distances['initial_leg_z'] = leg_pos[2].item()
            
        height_diff = leg_pos[2].item() - self.prev_distances['initial_leg_z']
        
        if height_diff > self.thresholds['pick_height']:
            return 1.0
        elif height_diff > 0:
            return height_diff / self.thresholds['pick_height']
        else:
            return 0.0
    
    def _transport_reward(self, leg_pos: torch.Tensor, target_pos: torch.Tensor, gripper_state: torch.Tensor) -> float:
        """Reward for transporting leg towards target."""
        # Handle both scalar and tensor gripper_state
        if isinstance(gripper_state, torch.Tensor):
            if gripper_state.numel() == 1:
                grasp_closed = gripper_state.item() < 0.03
            else:
                grasp_closed = (gripper_state < 0.03).all()
        else:
            grasp_closed = gripper_state < 0.03
            
        if not grasp_closed:  # Must be grasping
            return 0.0
            
        dist_to_target = torch.norm(leg_pos - target_pos)
        
        # Progress-based reward
        if 'prev_transport_dist' in self.prev_distances:
            prev_dist = self.prev_distances['prev_transport_dist']
            progress = prev_dist - dist_to_target.item()
            reward = max(0.0, progress * 10.0)  # Scale progress
        else:
            reward = 0.0
            
        self.prev_distances['prev_transport_dist'] = dist_to_target.item()
        
        # Bonus for getting close
        if dist_to_target < self.thresholds['align_dist']:
            reward += 0.5
            
        return min(1.0, reward)
    
    def _alignment_reward(self, leg_pos: torch.Tensor, leg_quat: torch.Tensor, 
                         target_pos: torch.Tensor, target_quat: torch.Tensor, 
                         gripper_state: torch.Tensor) -> float:
        """Reward for fine alignment before assembly."""
        # Handle both scalar and tensor gripper_state
        if isinstance(gripper_state, torch.Tensor):
            if gripper_state.numel() == 1:
                grasp_closed = gripper_state.item() < 0.03
            else:
                grasp_closed = (gripper_state < 0.03).all()
        else:
            grasp_closed = gripper_state < 0.03
            
        if not grasp_closed:  # Must be grasping
            return 0.0
            
        pos_dist = torch.norm(leg_pos - target_pos)
        
        # Angular alignment
        dot_product = torch.sum(leg_quat * target_quat)
        angle_diff = 2 * torch.acos(torch.clamp(torch.abs(dot_product), 0, 1))
        
        pos_reward = max(0.0, 1.0 - pos_dist.item() / self.thresholds['align_dist'])
        angle_reward = max(0.0, 1.0 - angle_diff.item() / self.thresholds['align_angle'])
        
        return 0.5 * pos_reward + 0.5 * angle_reward
    
    def _screwing_reward(self, leg_pos: torch.Tensor, leg_quat: torch.Tensor,
                        target_pos: torch.Tensor, target_quat: torch.Tensor, 
                        gripper_state: torch.Tensor) -> float:
        """Reward for screwing motion - rotational movement when close to target."""
        # Handle both scalar and tensor gripper_state
        if isinstance(gripper_state, torch.Tensor):
            if gripper_state.numel() == 1:
                grasping = gripper_state.item() < 0.03
            else:
                grasping = (gripper_state < 0.03).all()
        else:
            grasping = gripper_state < 0.03
            
        if not grasping:  # Must be grasping to screw
            return 0.0
            
        pos_dist = torch.norm(leg_pos - target_pos)
        
        # Only give screwing rewards when very close to target position
        if pos_dist > self.thresholds['screw_dist']:
            return 0.0
            
        # Track rotational motion
        if self.prev_leg_quat is None:
            self.prev_leg_quat = leg_quat.clone()
            return 0.0
            
        # Calculate rotational change
        dot_product = torch.sum(self.prev_leg_quat * leg_quat)
        rotation_angle = 2 * torch.acos(torch.clamp(torch.abs(dot_product), 0, 1))
        
        # Reward for rotational motion (screwing)
        rotation_reward = min(1.0, rotation_angle.item() / self.thresholds['screw_angle'])
        
        # Check if rotation is towards correct orientation
        target_dot = torch.sum(leg_quat * target_quat)
        target_angle = 2 * torch.acos(torch.clamp(torch.abs(target_dot), 0, 1))
        
        prev_target_dot = torch.sum(self.prev_leg_quat * target_quat) 
        prev_target_angle = 2 * torch.acos(torch.clamp(torch.abs(prev_target_dot), 0, 1))
        
        # Bonus if rotation brings us closer to target orientation
        orientation_progress = prev_target_angle - target_angle
        progress_bonus = max(0.0, orientation_progress.item() * 5.0)  # Scale progress
        
        # Update previous quaternion for next step
        self.prev_leg_quat = leg_quat.clone()
        
        return min(1.0, rotation_reward + progress_bonus)
    
    def _assembly_reward(self, leg_pos: torch.Tensor, leg_quat: torch.Tensor,
                        target_pos: torch.Tensor, target_quat: torch.Tensor, 
                        assembled: bool) -> float:
        """Reward for final assembly."""
        if assembled:
            return 1.0
            
        pos_dist = torch.norm(leg_pos - target_pos)
        dot_product = torch.sum(leg_quat * target_quat)
        angle_diff = 2 * torch.acos(torch.clamp(torch.abs(dot_product), 0, 1))
        
        if (pos_dist < self.thresholds['assemble_dist'] and 
            angle_diff < self.thresholds['assemble_angle']):
            return 0.8  # Very close to assembly
        else:
            return 0.0
    
    def _efficiency_bonus(self) -> float:
        """Bonus for completing task efficiently."""
        # Diminishing bonus over time
        time_factor = max(0.0, 1.0 - self.step_count / 1000.0)
        return 0.1 * time_factor
    
    def _safety_penalty(self, gripper_pos: torch.Tensor, parts_poses: torch.Tensor) -> float:
        """Penalty for unsafe behaviors."""
        penalty = 0.0
        
        # Penalty for gripper going too low
        if gripper_pos[2] < 0.01:
            penalty += 0.5
            
        # Penalty for hitting table
        table_pos = parts_poses[0:3]
        if torch.norm(gripper_pos[:2] - table_pos[:2]) < 0.05 and gripper_pos[2] < table_pos[2] + 0.02:
            penalty += 0.3
            
        return -penalty
    
    def _update_stage_tracking(self, gripper_pos: torch.Tensor, leg_pos: torch.Tensor, 
                              target_pos: torch.Tensor, gripper_state: torch.Tensor, 
                              assembled: bool):
        """Update current stage based on task progress."""
        dist_to_leg = torch.norm(gripper_pos - leg_pos)
        dist_to_target = torch.norm(leg_pos - target_pos)
        
        # Handle both scalar and tensor gripper_state
        if isinstance(gripper_state, torch.Tensor):
            if gripper_state.numel() == 1:
                grasping = gripper_state.item() < 0.03
            else:
                grasping = (gripper_state < 0.03).all()
        else:
            grasping = gripper_state < 0.03
        
        # Initialize leg height tracking
        if 'initial_leg_z' not in self.prev_distances:
            self.prev_distances['initial_leg_z'] = leg_pos[2].item()
        
        current_height = leg_pos[2].item()
        initial_height = self.prev_distances['initial_leg_z']
        height_diff = current_height - initial_height
        
        # Debug info (can be removed later)
        print(f"Debug: grasping={grasping}, dist_to_target={dist_to_target:.3f}, assembled={assembled}")
        print(f"Debug: align_dist={self.thresholds['align_dist']}, screw_dist={self.thresholds['screw_dist']}")
        
        # Stage progression logic - allow forward progression, prevent regression
        if assembled:
            self.current_stage = 'assemble'
            self.stage_completion['assemble'] = True
        elif dist_to_target < self.thresholds['screw_dist'] and self.stage_completion.get('transport', False):
            # Start screwing when close to target and transport is complete
            self.current_stage = 'screw'
            self.stage_completion['screw'] = True
        elif dist_to_target < self.thresholds['align_dist'] and self.stage_completion.get('transport', False):
            # Start aligning when close to target and transport is complete
            self.current_stage = 'align'
            self.stage_completion['align'] = True
        elif self.stage_completion.get('pick', False) and dist_to_target < self.thresholds['transport_dist']:
            # Continue transport if pick is complete
            self.current_stage = 'transport'
            self.stage_completion['transport'] = True
        elif grasping and height_diff > self.thresholds['pick_height']:
            self.current_stage = 'pick'
            self.stage_completion['pick'] = True
        elif grasping:  # If grasping, we're at least in grasp stage
            self.current_stage = 'grasp'
            self.stage_completion['grasp'] = True
        elif dist_to_leg < self.thresholds['grasp_dist']:  # Close to leg - grasp stage
            self.current_stage = 'grasp'
            self.stage_completion['grasp'] = True
        elif dist_to_leg < self.thresholds['approach_dist']:  # Approaching
            self.current_stage = 'approach'
            self.stage_completion['approach'] = True
        else:
            self.current_stage = 'approach'
            
        # Maintain highest achieved stage (no regression)
        stage_order = ['approach', 'grasp', 'pick', 'transport', 'align', 'screw', 'assemble']
        current_stage_idx = stage_order.index(self.current_stage)
        completed_stages = [stage for stage, completed in self.stage_completion.items() if completed]
        if completed_stages:
            max_completed_idx = max(stage_order.index(stage) for stage in completed_stages)
            if max_completed_idx > current_stage_idx:
                self.current_stage = stage_order[max_completed_idx]