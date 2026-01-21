"""Piper cloth folding data collection with workspace-aware grasp positioning.

This script implements a complete cloth folding sequence for the Piper robot,
accounting for its limited reach (~350mm) by offsetting grasp positions inward
toward the cloth center.

State machine: APPROACH -> SETTLE -> GRASP -> LIFT -> FOLD -> RELEASE -> RETREAT -> DONE

Key fixes from original:
1. Grasp positions offset 3cm inward toward cloth center (Y~0.145m instead of 0.175m)
2. Relaxed tolerances (2.5cm instead of 2cm)
3. Complete fold cycle with FOLD, RELEASE, RETREAT phases
4. Uses grasp_assist with manual gripper signal injection

Usage:
    python collect_piper_cloth_fold.py --render --debug           # Demo mode
    python collect_piper_cloth_fold.py --episodes 50              # Data collection
    python collect_piper_cloth_fold.py --render --record          # Record video
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from collect_robosuite import create_bimanual_env


def create_video_writer(output_path: str, fps: int = 30, frame_size: Tuple[int, int] = (512, 512)):
    """Create an OpenCV video writer.

    Args:
        output_path: Path to output video file
        fps: Frames per second
        frame_size: (width, height) of video frames

    Returns:
        cv2.VideoWriter object or None if OpenCV not available
    """
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), frame_size)
        if writer.isOpened():
            print(f"Video writer opened: {output_path}")
            return writer
        else:
            print(f"Warning: Could not open video writer for {output_path}")
            return None
    except ImportError:
        print("Warning: OpenCV not installed. Video recording disabled.")
        print("Install with: pip install opencv-python")
        return None


def add_frame_to_video(writer, frame: np.ndarray):
    """Add a frame to the video writer.

    Args:
        writer: cv2.VideoWriter object
        frame: RGB image array (H, W, 3) uint8
    """
    if writer is None:
        return
    if frame is None or frame.size == 0:
        return
    try:
        import cv2
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    except Exception as e:
        print(f"Warning: Could not write frame: {e}")


class FoldPhase(Enum):
    """State machine phases for cloth folding."""
    APPROACH = "approach"    # Move to above grasp positions
    SETTLE = "settle"        # Stabilize at approach height
    DESCEND = "descend"      # NEW: Slow descent with gripper closing (top-down approach)
    GRASP = "grasp"          # Dwell at grasp position to establish friction contact
    LIFT = "lift"            # Raise cloth edge
    FOLD = "fold"            # Move toward opposite side
    RELEASE = "release"      # Open grippers
    RETREAT = "retreat"      # Move up and away
    DONE = "done"            # Complete


@dataclass
class PiperClothFoldConfig:
    """Configuration for Piper cloth folding with workspace-aware parameters.

    Piper Robot Setup:
    - Base positions: Y=±0.16m (parallel configuration)
    - Joint1 rotations: [0.15, -0.35] (points arms inward toward cloth)
    - EEF start positions: ~Y=±0.11m (already inward of cloth corners)
    - Cloth corners at Y=±0.15m (narrower 30cm cloth for reachability)
    - Grasp targets at Y=±0.14m (1cm inward offset from corners)

    Top-Down Approach: Start high, descend slowly while closing gripper,
    press cloth against table to establish friction grip.

    Both arms can reach their targets within 2.5cm tolerance.
    """
    # Heights relative to cloth (cloth at Z=0.81)
    cloth_z: float = 0.81
    approach_height: float = 0.06       # 6cm above cloth for top-down approach (was 0.03)
    descend_height: float = 0.02        # Height to start closing gripper during descent
    grasp_height: float = 0.015         # 15mm above cloth (25mm above table to clear gripper fingers)
    lift_height: float = 0.10           # Lift to 10cm above cloth (safe margin for cloth sag)
    fold_height: float = 0.08           # Height during fold motion (8cm for clearance)
    release_height: float = 0.02        # Height when releasing
    retreat_height: float = 0.10        # Final retreat height

    # Inward offset toward cloth center (brings targets into workspace)
    # With bases at Y=±0.16 and joint1 rotations, EEF starts at ~Y=±0.11
    # Cloth corners at Y=±0.15, grasp targets at Y=±0.11 with 4cm offset
    # Larger offset gives more kinematic slack for vertical (Z) descent
    grasp_inward_offset: float = 0.04   # 4cm inward from corners (was 0.01)

    # Relaxed tolerances (increased from problematic 2cm)
    position_tolerance: float = 0.025   # 2.5cm for phase transitions
    grasp_dist_threshold: float = 0.06  # 6cm for approach phase

    # Phase timing (min/max steps)
    # Reduced max_steps for position-based transitions (early exit on convergence)
    min_approach_steps: int = 30   # Minimum for smooth approach
    max_approach_steps: int = 150  # Reduced from 200
    min_settle_steps: int = 25     # More settling reduces oscillation
    max_settle_steps: int = 40     # Reduced from 60
    min_descend_steps: int = 25    # Slow descent for contact
    max_descend_steps: int = 40    # Reduced from 60
    min_grasp_steps: int = 30      # Dwell to establish grip
    max_grasp_steps: int = 40      # Reduced from 60
    min_lift_steps: int = 30       # Slower lift for stable grip
    max_lift_steps: int = 60       # Reduced from 80
    min_fold_steps: int = 30
    max_fold_steps: int = 60       # Reduced from 100 - major time savings
    min_release_steps: int = 10
    max_release_steps: int = 20    # Reduced from 30
    min_retreat_steps: int = 15
    max_retreat_steps: int = 30    # Reduced from 50

    # OSC controller parameters
    osc_output_max: float = 0.05        # OSC position output scale
    pos_gain: float = 1.0               # Position control gain

    # Gripper action values (for grasp_assist)
    gripper_close_action: float = 1.0
    gripper_open_action: float = -1.0

    # Fold target calculation
    fold_overshoot: float = 0.02        # 2cm past center (minimal overshoot)

    # Dynamic cloth tracking parameters
    tracking_enabled: bool = True       # Enable dynamic tracking during approach
    target_smoothing: float = 0.2       # Smoothing factor for target updates (0=none, 1=frozen)
    lock_on_grasp: bool = True          # Lock targets when GRASP phase begins

    # Smooth gripper parameters
    gripper_close_duration: int = 15    # Steps to fully close gripper (~0.3s at 50Hz)
    gripper_open_duration: int = 10     # Steps to fully open gripper
    gripper_open_pos: float = 0.030     # joint7 position when open
    gripper_closed_pos: float = 0.005   # joint7 position when closed

    # Tool offset: distance from EEF (link6) to gripper fingers (link7/8)
    # At initial pose, link6 is at X=-0.505 and link7/8 at X=-0.370 (0.135m ahead)
    # When EEF is commanded to cloth corner, fingers end up 0.135m past it (in middle!)
    # We offset EEF target backward so fingers land at the actual corner.
    # Negative value = EEF target moves back, fingers move back toward corner
    tool_offset_x: float = -0.10        # Offset EEF target back so fingers reach corners


def compute_grasp_positions(
    corners: np.ndarray,
    config: PiperClothFoldConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute workspace-aware grasp positions with inward offset.

    With robot bases at Y=±0.16m and joint1 rotations pointing inward,
    EEF starts at ~Y=±0.11m. Cloth corners at Y=±0.15m, so targets at
    Y=±0.14m (with 1cm inward offset) are well within reach.

    Args:
        corners: (4, 3) cloth corner positions
        config: Configuration with grasp_inward_offset

    Returns:
        grasp_pos0, grasp_pos1: Grasp positions for robot0 (Y<0) and robot1 (Y>0)
    """
    # Find left corners (smallest X values) for parallel layout
    xs = corners[:, 0]
    left_indices = np.argsort(xs)[:2]
    left_corners = corners[left_indices]

    # Sort by Y: robot0 handles negative Y, robot1 handles positive Y
    if left_corners[0, 1] < left_corners[1, 1]:
        corner0 = left_corners[0].copy()  # Bottom-left (Y < 0) -> robot0
        corner1 = left_corners[1].copy()  # Top-left (Y > 0) -> robot1
    else:
        corner0 = left_corners[1].copy()
        corner1 = left_corners[0].copy()

    # Calculate cloth center
    cloth_center = corners.mean(axis=0)

    # Compute direction from each corner toward center (XY only)
    dir0 = cloth_center[:2] - corner0[:2]
    dir0 = dir0 / (np.linalg.norm(dir0) + 1e-6)

    dir1 = cloth_center[:2] - corner1[:2]
    dir1 = dir1 / (np.linalg.norm(dir1) + 1e-6)

    # Apply inward offset (THE KEY FIX)
    grasp_pos0 = corner0.copy()
    grasp_pos0[:2] += dir0 * config.grasp_inward_offset

    grasp_pos1 = corner1.copy()
    grasp_pos1[:2] += dir1 * config.grasp_inward_offset

    # Apply tool offset to compensate for EEF-to-gripper-finger distance
    # The EEF (link6) is behind the gripper fingers (link7/8) in the robot's local X
    # When we command EEF to a position, the gripper fingers end up further in +X (world)
    # So we offset the EEF target by -X to make the fingers land at the actual grasp position
    # Note: This assumes robots are facing +X direction (toward the cloth)
    grasp_pos0[0] += config.tool_offset_x
    grasp_pos1[0] += config.tool_offset_x

    return grasp_pos0, grasp_pos1


def compute_fold_targets(
    grasp_pos0: np.ndarray,
    grasp_pos1: np.ndarray,
    corners: np.ndarray,
    config: PiperClothFoldConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fold targets - move left edge toward right side.

    For parallel layout: fold along X axis (left -> right)
    """
    # Find cloth center X and compute fold target
    cloth_center_x = corners[:, 0].mean()
    fold_target_x = cloth_center_x + config.fold_overshoot

    # Keep Y positions same, use fold height
    target0 = np.array([
        fold_target_x,
        grasp_pos0[1],
        config.cloth_z + config.fold_height
    ])
    target1 = np.array([
        fold_target_x,
        grasp_pos1[1],
        config.cloth_z + config.fold_height
    ])

    return target0, target1


def check_phase_complete(
    eef0: np.ndarray,
    target0: np.ndarray,
    eef1: np.ndarray,
    target1: np.ndarray,
    phase_step: int,
    min_steps: int,
    max_steps: int,
    tolerance: float,
    debug: bool = False,
    phase_name: str = "",
) -> Tuple[bool, str]:
    """Check if phase is complete based on convergence and step counts.

    Returns:
        (is_complete, reason)
    """
    dist0 = np.linalg.norm(target0 - eef0)
    dist1 = np.linalg.norm(target1 - eef1)
    converged = dist0 < tolerance and dist1 < tolerance

    # Max steps reached - force transition
    if phase_step >= max_steps:
        return True, "max_steps"

    # Converged after minimum steps - early exit
    if converged and phase_step >= min_steps:
        if debug:
            print(f"  [Early exit] {phase_name} at step {phase_step} "
                  f"(dist0={dist0:.4f}, dist1={dist1:.4f})")
        return True, "converged"

    return False, ""


def compute_action(
    eef0: np.ndarray,
    target0: np.ndarray,
    eef1: np.ndarray,
    target1: np.ndarray,
    gripper_action: float,
    config: PiperClothFoldConfig,
    action_dim: int = 6,
) -> np.ndarray:
    """Compute action for bimanual OSC controller.

    Supports multiple action formats:
    - action_dim=6:  [pos0(3), pos1(3)] - OSC_POSITION with no gripper DOF
    - action_dim=12: [pos0(3), ori0(3), pos1(3), ori1(3)] - OSC_POSE with no gripper
    - action_dim=14: [pos0(3), ori0(3), grip0(1), pos1(3), ori1(3), grip1(1)]

    The gripper_action is used by grasp_assist logic in the environment
    to determine when to pin cloth vertices.

    Returns:
        action: Array of appropriate dimension
    """
    # Position deltas
    delta0 = config.pos_gain * (target0 - eef0)
    delta1 = config.pos_gain * (target1 - eef1)

    # Normalize for OSC controller
    action0_pos = np.clip(delta0 / config.osc_output_max, -1.0, 1.0)
    action1_pos = np.clip(delta1 / config.osc_output_max, -1.0, 1.0)

    if action_dim == 6:
        # OSC_POSITION with no gripper DOF: 3 per arm
        # Format: [pos0(3), pos1(3)]
        action = np.concatenate([
            action0_pos,           # Robot0 position (3)
            action1_pos,           # Robot1 position (3)
        ])
    elif action_dim == 12:
        # OSC_POSE with no gripper DOF: 6 per arm
        # Format: [pos0(3), ori0(3), pos1(3), ori1(3)]
        action = np.concatenate([
            action0_pos,           # Robot0 position (3)
            np.zeros(3),           # Robot0 orientation (3)
            action1_pos,           # Robot1 position (3)
            np.zeros(3),           # Robot1 orientation (3)
        ])
    else:
        # Standard with gripper control: 14-dim
        action = np.concatenate([
            action0_pos,           # Robot0 position (3)
            np.zeros(3),           # Robot0 orientation (3)
            [gripper_action],      # Robot0 gripper (1)
            action1_pos,           # Robot1 position (3)
            np.zeros(3),           # Robot1 orientation (3)
            [gripper_action],      # Robot1 gripper (1)
        ])

    return action


class PiperClothFoldPolicy:
    """Scripted cloth folding policy for Piper robot.

    Implements workspace-aware grasp positioning to account for Piper's
    limited reach (~350mm). The key fix is offsetting grasp positions
    3cm inward toward the cloth center.

    State machine: APPROACH -> SETTLE -> GRASP -> LIFT -> FOLD -> RELEASE -> RETREAT -> DONE
    """

    def __init__(
        self,
        config: Optional[PiperClothFoldConfig] = None,
        debug: bool = False,
        action_dim: int = 12,
        action_smoothing: float = 0.3,  # Smoothing factor: 0.3 = 30% new, 70% old
    ):
        self.config = config or PiperClothFoldConfig()
        self.debug = debug
        self.action_dim = action_dim
        self._action_smoothing = action_smoothing
        self._current_gripper_signal = -1.0  # Track current gripper state for grasp_assist
        self.reset()

    def reset(self):
        """Reset policy state for new episode."""
        self.phase = FoldPhase.APPROACH
        self.phase_step = 0
        self._grasp_pos0 = None
        self._grasp_pos1 = None
        self._fold_target0 = None
        self._fold_target1 = None
        self._done = False
        self._current_gripper_signal = -1.0
        self._prev_action = None  # For action smoothing
        self._tracking_locked = False  # Whether cloth tracking is locked

        # Smooth gripper state
        self._gripper_current = self.config.gripper_open_pos
        self._gripper_target = self.config.gripper_open_pos
        self._gripper_start = self.config.gripper_open_pos
        self._gripper_step = 0

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to reduce jerky movements."""
        if self._prev_action is None:
            self._prev_action = action.copy()
            return action
        smoothed = self._action_smoothing * action + (1 - self._action_smoothing) * self._prev_action
        self._prev_action = smoothed.copy()
        return smoothed

    def _update_gripper_position(self, closing: bool):
        """Update gripper position with smooth ramping.

        Args:
            closing: True if gripper should be closing, False if opening
        """
        target = self.config.gripper_closed_pos if closing else self.config.gripper_open_pos
        duration = (self.config.gripper_close_duration if closing
                    else self.config.gripper_open_duration)

        # Detect target change - start new transition
        if target != self._gripper_target:
            self._gripper_target = target
            self._gripper_start = self._gripper_current
            self._gripper_step = 0

        # Linear interpolation toward target
        if self._gripper_step >= duration:
            self._gripper_current = self._gripper_target
        else:
            progress = self._gripper_step / duration
            self._gripper_current = (self._gripper_start +
                                     progress * (self._gripper_target - self._gripper_start))
            self._gripper_step += 1

    @property
    def gripper_position(self) -> float:
        """Current gripper joint7 position for qpos control."""
        return self._gripper_current

    @property
    def gripper_signal(self) -> float:
        """Get current gripper signal for grasp_assist."""
        return self._current_gripper_signal

    @property
    def done(self) -> bool:
        """Returns True when the policy has completed all phases."""
        return self._done

    def _transition_to(self, new_phase: FoldPhase):
        """Transition to a new phase."""
        if self.debug:
            print(f"Phase transition: {self.phase.value} -> {new_phase.value} "
                  f"at step {self.phase_step}")

        # Lock cloth tracking when entering DESCEND phase
        # This prevents tracking deformed cloth corners once we start descending
        if new_phase == FoldPhase.DESCEND and self.config.lock_on_grasp:
            self._tracking_locked = True
            if self.debug:
                print(f"  [Tracking locked] Final grasp targets:")
                print(f"    pos0={self._grasp_pos0}, pos1={self._grasp_pos1}")

        self.phase = new_phase
        self.phase_step = 0

    def predict(self, obs: dict) -> np.ndarray:
        """Generate action based on current observation and phase."""
        eef0 = np.asarray(obs["robot0_eef_pos"])
        eef1 = np.asarray(obs["robot1_eef_pos"])
        corners = np.asarray(obs["cloth_corners"]).reshape(4, 3)

        # Dynamic cloth tracking during APPROACH and SETTLE phases
        # Once DESCEND begins, targets are locked to prevent tracking deformed cloth
        should_track = (self.config.tracking_enabled and
                        not self._tracking_locked and
                        self.phase in [FoldPhase.APPROACH, FoldPhase.SETTLE])

        if self._grasp_pos0 is None or should_track:
            # Compute raw grasp positions from current cloth corners
            raw_grasp0, raw_grasp1 = compute_grasp_positions(corners, self.config)

            # Apply smoothing to prevent jittery targets from cloth simulation noise
            if self._grasp_pos0 is not None and self.config.target_smoothing > 0:
                alpha = self.config.target_smoothing
                self._grasp_pos0 = alpha * self._grasp_pos0 + (1 - alpha) * raw_grasp0
                self._grasp_pos1 = alpha * self._grasp_pos1 + (1 - alpha) * raw_grasp1
            else:
                self._grasp_pos0 = raw_grasp0
                self._grasp_pos1 = raw_grasp1

            # Update fold targets during tracking
            self._fold_target0, self._fold_target1 = compute_fold_targets(
                self._grasp_pos0, self._grasp_pos1, corners, self.config
            )

            if self.debug and (self._grasp_pos0 is None or self.phase_step % 50 == 0):
                print(f"\n[Tracking] Grasp positions updated (phase={self.phase.value}):")
                print(f"  Cloth corners (left): {corners[np.argsort(corners[:,0])[:2]]}")
                print(f"  Grasp pos0 (robot0): {self._grasp_pos0}")
                print(f"  Grasp pos1 (robot1): {self._grasp_pos1}")

        # Phase-specific logic
        if self.phase == FoldPhase.APPROACH:
            target0 = self._grasp_pos0.copy()
            target0[2] = self.config.cloth_z + self.config.approach_height
            target1 = self._grasp_pos1.copy()
            target1[2] = self.config.cloth_z + self.config.approach_height
            gripper = self.config.gripper_open_action

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_approach_steps, self.config.max_approach_steps,
                self.config.grasp_dist_threshold,  # Use larger threshold for approach
                self.debug, "APPROACH"
            )
            if complete:
                self._transition_to(FoldPhase.SETTLE)

        elif self.phase == FoldPhase.SETTLE:
            target0 = self._grasp_pos0.copy()
            target0[2] = self.config.cloth_z + self.config.approach_height
            target1 = self._grasp_pos1.copy()
            target1[2] = self.config.cloth_z + self.config.approach_height
            gripper = self.config.gripper_open_action

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_settle_steps, self.config.max_settle_steps,
                self.config.position_tolerance,
                self.debug, "SETTLE"
            )
            if complete:
                self._transition_to(FoldPhase.DESCEND)

        elif self.phase == FoldPhase.DESCEND:
            # Top-down approach: slowly descend while gradually closing gripper
            # This allows gripper to contact cloth and press it against the table
            target0 = self._grasp_pos0.copy()
            target0[2] = self.config.cloth_z + self.config.grasp_height  # Target is at/below cloth
            target1 = self._grasp_pos1.copy()
            target1[2] = self.config.cloth_z + self.config.grasp_height

            # Gradually close gripper during descent (from open to close over descend phase)
            gripper_progress = min(1.0, self.phase_step / max(1, self.config.max_descend_steps * 0.7))
            gripper = self.config.gripper_open_action + gripper_progress * (
                self.config.gripper_close_action - self.config.gripper_open_action
            )

            if self.debug and self.phase_step % 10 == 0:
                print(f"  [DESCEND] Step {self.phase_step}: gripper={gripper:.2f}, "
                      f"eef0_z={eef0[2]:.4f}, target_z={target0[2]:.4f}")

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_descend_steps, self.config.max_descend_steps,
                self.config.position_tolerance,
                self.debug, "DESCEND"
            )
            if complete:
                self._transition_to(FoldPhase.GRASP)

        elif self.phase == FoldPhase.GRASP:
            # Dwell at grasp position with closed gripper to establish friction contact
            # Gripper is pressing cloth against table - hold to let friction stabilize
            target0 = self._grasp_pos0.copy()
            target0[2] = self.config.cloth_z + self.config.grasp_height
            target1 = self._grasp_pos1.copy()
            target1[2] = self.config.cloth_z + self.config.grasp_height
            gripper = self.config.gripper_close_action  # Keep gripper closed

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_grasp_steps, self.config.max_grasp_steps,
                self.config.position_tolerance,
                self.debug, "GRASP"
            )
            if complete:
                self._transition_to(FoldPhase.LIFT)

        elif self.phase == FoldPhase.LIFT:
            target0 = self._grasp_pos0.copy()
            target0[2] = self.config.cloth_z + self.config.lift_height
            target1 = self._grasp_pos1.copy()
            target1[2] = self.config.cloth_z + self.config.lift_height
            gripper = self.config.gripper_close_action

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_lift_steps, self.config.max_lift_steps,
                self.config.position_tolerance,
                self.debug, "LIFT"
            )
            if complete:
                self._transition_to(FoldPhase.FOLD)

        elif self.phase == FoldPhase.FOLD:
            target0 = self._fold_target0.copy()
            target1 = self._fold_target1.copy()
            gripper = self.config.gripper_close_action

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_fold_steps, self.config.max_fold_steps,
                self.config.position_tolerance,
                self.debug, "FOLD"
            )
            if complete:
                self._transition_to(FoldPhase.RELEASE)

        elif self.phase == FoldPhase.RELEASE:
            target0 = self._fold_target0.copy()
            target0[2] = self.config.cloth_z + self.config.release_height
            target1 = self._fold_target1.copy()
            target1[2] = self.config.cloth_z + self.config.release_height
            gripper = self.config.gripper_open_action  # Open gripper to release

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_release_steps, self.config.max_release_steps,
                self.config.position_tolerance,
                self.debug, "RELEASE"
            )
            if complete:
                self._transition_to(FoldPhase.RETREAT)

        elif self.phase == FoldPhase.RETREAT:
            target0 = self._fold_target0.copy()
            target0[2] = self.config.cloth_z + self.config.retreat_height
            target1 = self._fold_target1.copy()
            target1[2] = self.config.cloth_z + self.config.retreat_height
            gripper = self.config.gripper_open_action

            complete, _ = check_phase_complete(
                eef0, target0, eef1, target1, self.phase_step,
                self.config.min_retreat_steps, self.config.max_retreat_steps,
                self.config.position_tolerance,
                self.debug, "RETREAT"
            )
            if complete:
                self._transition_to(FoldPhase.DONE)
                self._done = True

        else:  # DONE - hold position
            target0 = self._fold_target0.copy()
            target0[2] = self.config.cloth_z + self.config.retreat_height
            target1 = self._fold_target1.copy()
            target1[2] = self.config.cloth_z + self.config.retreat_height
            gripper = self.config.gripper_open_action

        self.phase_step += 1
        self._current_gripper_signal = gripper

        # Update smooth gripper position
        closing = self.phase in [FoldPhase.GRASP, FoldPhase.LIFT, FoldPhase.FOLD]
        self._update_gripper_position(closing)

        raw_action = compute_action(eef0, target0, eef1, target1, gripper, self.config, self.action_dim)

        # Debug output
        if self.debug and self.phase_step % 20 == 0:
            dist0 = np.linalg.norm(target0 - eef0)
            dist1 = np.linalg.norm(target1 - eef1)
            print(f"[{self.phase.value:10s}] step={self.phase_step:3d} "
                  f"dist0={dist0:.4f} dist1={dist1:.4f} "
                  f"z0={eef0[2]:.3f} z1={eef1[2]:.3f} "
                  f"grip_sig={gripper:.1f} grip_pos={self._gripper_current:.4f}")
            # Print detailed positions and actions
            print(f"  EEF0: X={eef0[0]:.3f} Y={eef0[1]:.3f} Z={eef0[2]:.3f}")
            print(f"  EEF1: X={eef1[0]:.3f} Y={eef1[1]:.3f} Z={eef1[2]:.3f}")
            print(f"  TGT0: X={target0[0]:.3f} Y={target0[1]:.3f} Z={target0[2]:.3f}")
            print(f"  TGT1: X={target1[0]:.3f} Y={target1[1]:.3f} Z={target1[2]:.3f}")
            print(f"  ACT0: X={raw_action[0]:.3f} Y={raw_action[1]:.3f} Z={raw_action[2]:.3f}")
            print(f"  ACT1: X={raw_action[6]:.3f} Y={raw_action[7]:.3f} Z={raw_action[8]:.3f}")

        return self._smooth_action(raw_action)


def main():
    parser = argparse.ArgumentParser(
        description="Piper cloth folding with workspace-aware grasp positioning"
    )
    parser.add_argument("--render", action="store_true", help="Show visualization")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode without data collection")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes (default: 1 for demo)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--cloth_preset", type=str, default="piper",
                        help="Cloth configuration preset")
    parser.add_argument("--record", action="store_true",
                        help="Record video of the episode")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Output path for recorded video (default: auto-generated)")
    parser.add_argument("--record_fps", type=int, default=30,
                        help="FPS for recorded video (default: 30)")
    args = parser.parse_args()

    # Recording requires rendering
    if args.record and not args.render:
        print("Note: --record requires --render, enabling rendering...")
        args.render = True

    print("\n" + "=" * 70)
    print(" PIPER CLOTH FOLD - Workspace-Aware Grasp Positioning")
    print("=" * 70)
    print("\nSetup: Robot bases at Y=±0.16m, joint1 rotations point arms inward")
    print("EEF starts at ~Y=±0.11m, cloth corners at Y=±0.15m, targets at Y=±0.14m")
    print("\nCreating environment (this may take a moment)...", flush=True)

    # Create environment with strict grasp-assist enabled
    # Pins cloth vertices to gripper when gripper closes near cloth
    # Strict mode uses tight tolerances: 2cm vertical, 3cm horizontal
    # cloth_x_offset=-0.05 moves cloth closer to robots for better kinematic reach
    env = create_bimanual_env(
        robots="Piper",
        task="TwoArmClothFold",
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        cloth_preset=args.cloth_preset,
        grasp_assist=True,  # Enable grasp-assist for reliable cloth pickup
        assist_strict=True,  # Use strict detection with tight tolerances
        assist_z_tolerance=0.02,  # 2cm vertical tolerance
        assist_xy_radius=0.03,  # 3cm horizontal radius
        ignore_done=True,
        cloth_x_offset=-0.05,  # Move cloth closer to robots for better Z reach
    )
    print("Environment created!")
    print(f"Action dimension: {env.action_dim}")

    # List available cameras for debugging
    if args.record or args.debug:
        cam_names = [env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)]
        print(f"Available cameras: {cam_names}")

    # Create policy with correct action dimension
    config = PiperClothFoldConfig()
    policy = PiperClothFoldPolicy(config, debug=args.debug, action_dim=env.action_dim)

    # Run episodes
    for episode in range(args.episodes):
        print(f"\n{'='*70}")
        print(f" Episode {episode + 1}/{args.episodes}")
        print(f"{'='*70}")

        # Setup video recording for this episode
        video_writer = None
        if args.record:
            if args.record_path:
                video_path = args.record_path
            else:
                # Auto-generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_dir = Path("recordings")
                video_dir.mkdir(exist_ok=True)
                video_path = str(video_dir / f"piper_cloth_fold_{timestamp}_ep{episode+1}.mp4")
            print(f"Recording to: {video_path}")
            video_writer = create_video_writer(video_path, fps=args.record_fps, frame_size=(512, 512))

        obs = env.reset()
        policy.reset()

        # Print initial cloth and robot positions
        corners = obs["cloth_corners"].reshape(4, 3)
        eef0 = obs["robot0_eef_pos"]
        eef1 = obs["robot1_eef_pos"]

        print("\nInitial state:")
        print(f"  Cloth corners (left edge Y): {corners[np.argsort(corners[:,0])[:2], 1]}")
        print(f"  Robot0 EEF: {eef0}")
        print(f"  Robot1 EEF: {eef1}")

        total_steps = 0
        success = False
        frames_recorded = 0

        for step in range(args.max_steps):
            action = policy.predict(obs)

            # Set gripper signals for grasp_assist (required for 0-DOF grippers)
            # Both grippers use the same signal (closing or opening together)
            env.gripper_signals = [policy.gripper_signal, policy.gripper_signal]

            obs, reward, done, info = env.step(action)

            # AFTER step: Directly control gripper joint positions using smooth ramping
            # The Piper gripper has dof=0 so no controller actuates it - physics can cause drift
            # joint7: 0 = closed, 0.035 = open (for each arm)
            # joint8 is coupled: joint8 = -joint7
            # Use smooth gripper position from policy (gradual close/open over multiple steps)
            target_j7 = policy.gripper_position

            # Set joint qpos for both robot grippers AFTER physics step
            # Robot 0: qpos[6]=joint7, qpos[7]=joint8
            # Robot 1: qpos[14]=joint7, qpos[15]=joint8
            try:
                # Robot 0 gripper
                env.sim.data.qpos[6] = target_j7
                env.sim.data.qpos[7] = -target_j7
                # Robot 1 gripper
                env.sim.data.qpos[14] = target_j7
                env.sim.data.qpos[15] = -target_j7
                # Zero velocities to prevent oscillation
                env.sim.data.qvel[6] = 0.0
                env.sim.data.qvel[7] = 0.0
                env.sim.data.qvel[14] = 0.0
                env.sim.data.qvel[15] = 0.0
                # Recompute physics state with new positions
                env.sim.forward()
            except (IndexError, AttributeError):
                pass  # Skip if indices don't match

            if args.render:
                env.render()

            # Capture frame for video recording
            if args.record and video_writer is not None:
                # Use MuJoCo's offscreen rendering with bimanual_view camera
                frame = env.sim.render(
                    width=512,
                    height=512,
                    camera_name="bimanual_view",
                    mode="offscreen",
                )
                # Frame comes as (H, W, 3) RGB, flip vertically (MuJoCo convention)
                if frame is not None and frame.size > 0:
                    frame = np.flipud(frame).copy()  # .copy() ensures contiguous array
                    add_frame_to_video(video_writer, frame)
                    frames_recorded += 1

            # Debug logging for gripper and cloth state
            if args.debug and step % 10 == 0:
                # Gripper state
                g0_qpos = obs.get("robot0_gripper_qpos", [])
                g1_qpos = obs.get("robot1_gripper_qpos", [])
                print(f"  Gripper qpos: r0={g0_qpos}, r1={g1_qpos}")

                # Gripper position (physics-based control)
                print(f"  Gripper target: {policy.gripper_position:.4f}")

                # EEF Z positions vs cloth Z (for contact verification)
                eef0_z = obs["robot0_eef_pos"][2]
                eef1_z = obs["robot1_eef_pos"][2]
                cloth_z = 0.81
                print(f"  EEF Z: r0={eef0_z:.3f}, r1={eef1_z:.3f}, cloth={cloth_z:.3f}")
                print(f"  Z distance from cloth: r0={abs(eef0_z-cloth_z):.3f}, r1={abs(eef1_z-cloth_z):.3f}")

                # Cloth center position (to verify if cloth is moving with gripper)
                cloth_center = obs["cloth_center"]
                print(f"  Cloth center: Z={cloth_center[2]:.3f}")

            total_steps += 1

            if policy.done:
                success = True
                break

        # Check final state
        final_corners = obs["cloth_corners"].reshape(4, 3)
        initial_left_x = corners[:, 0].min()
        final_left_x = final_corners[:, 0].min()
        cloth_moved = final_left_x - initial_left_x

        print("\nEpisode complete:")
        print(f"  Total steps: {total_steps}")
        print(f"  Policy done: {policy.done}")
        print(f"  Initial corners X: {corners[:, 0]}")
        print(f"  Final corners X: {final_corners[:, 0]}")
        print(f"  Cloth X movement (left edge): {cloth_moved:.3f}m")
        print(f"  Success: {success and cloth_moved > 0.03}")

        # Hold final position briefly
        if args.render:
            print("\nHolding final position...")
            for _ in range(50):
                action = np.zeros(env.action_dim)
                obs, _, _, _ = env.step(action)
                env.render()
                # Continue recording during hold
                if args.record and video_writer is not None:
                    frame = env.sim.render(width=512, height=512, camera_name="bimanual_view", mode="offscreen")
                    if frame is not None and frame.size > 0:
                        frame = np.flipud(frame).copy()
                        add_frame_to_video(video_writer, frame)

        # Close video writer for this episode
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path} ({frames_recorded} frames)")

    print("\nDone!")
    env.close()


if __name__ == "__main__":
    main()
