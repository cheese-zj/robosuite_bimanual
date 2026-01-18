"""Scripted policies for bimanual robosuite tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class ScriptedPolicyConfig:
    approach_height: float = 0.06
    grasp_height: float = 0.01
    lift_height: float = 0.18
    pos_gain: float = 1.0
    max_delta: float = 0.05
    grasp_dist_threshold: float = 0.025
    settle_steps: int = 8
    grasp_steps: int = 12
    lift_steps: int = 40
    open_gripper: float = -1.0
    close_gripper: float = 1.0


class ScriptedBimanualPolicy:
    """Simple waypoint-based scripted policy for TwoArmLift-style tasks."""

    def __init__(self, config: Optional[ScriptedPolicyConfig] = None):
        self.config = config or ScriptedPolicyConfig()
        self.phase = "approach"
        self.phase_step = 0

    def reset(self):
        self.phase = "approach"
        self.phase_step = 0

    def _get_handle_positions(self, obs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        pot_pos = np.asarray(obs.get("pot_pos", np.zeros(3)))
        handle0 = obs.get("handle0_xpos")
        handle1 = obs.get("handle1_xpos")

        if handle0 is None:
            handle0 = pot_pos + np.array([-0.05, 0.0, 0.0])
        if handle1 is None:
            handle1 = pot_pos + np.array([0.05, 0.0, 0.0])

        return np.asarray(handle0), np.asarray(handle1)

    def _compute_arm_action(self, current: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        """Compute normalized action for OSC controller.

        The OSC controller expects normalized inputs [-1, 1] which it scales to output_max (0.05m).
        """
        delta = self.config.pos_gain * (target - current)
        # Normalize delta to [-1, 1] range for OSC controller
        osc_output_max = 0.05
        normalized_delta = delta / osc_output_max
        normalized_delta = np.clip(normalized_delta, -1.0, 1.0)
        return np.concatenate([normalized_delta, np.zeros(3), np.array([gripper])])

    def predict(self, obs: Dict) -> np.ndarray:
        eef0 = obs.get("robot0_eef_pos")
        eef1 = obs.get("robot1_eef_pos")
        if eef0 is None or eef1 is None:
            raise ValueError("Observation is missing robot eef positions")

        handle0, handle1 = self._get_handle_positions(obs)

        if self.phase == "approach":
            target0 = handle0 + np.array([0.0, 0.0, self.config.approach_height])
            target1 = handle1 + np.array([0.0, 0.0, self.config.approach_height])
            gripper = self.config.open_gripper

            dist0 = np.linalg.norm(target0 - eef0)
            dist1 = np.linalg.norm(target1 - eef1)
            if dist0 < self.config.grasp_dist_threshold and dist1 < self.config.grasp_dist_threshold:
                if self.phase_step >= self.config.settle_steps:
                    self.phase = "grasp"
                    self.phase_step = 0

        elif self.phase == "grasp":
            target0 = handle0 + np.array([0.0, 0.0, self.config.grasp_height])
            target1 = handle1 + np.array([0.0, 0.0, self.config.grasp_height])
            gripper = self.config.close_gripper

            if self.phase_step >= self.config.grasp_steps:
                self.phase = "lift"
                self.phase_step = 0

        elif self.phase == "lift":
            target0 = handle0 + np.array([0.0, 0.0, self.config.lift_height])
            target1 = handle1 + np.array([0.0, 0.0, self.config.lift_height])
            gripper = self.config.close_gripper

            if self.phase_step >= self.config.lift_steps:
                self.phase = "hold"
                self.phase_step = 0

        else:
            target0 = handle0 + np.array([0.0, 0.0, self.config.lift_height])
            target1 = handle1 + np.array([0.0, 0.0, self.config.lift_height])
            gripper = self.config.close_gripper

        action0 = self._compute_arm_action(np.asarray(eef0), target0, gripper)
        action1 = self._compute_arm_action(np.asarray(eef1), target1, gripper)

        self.phase_step += 1

        return np.concatenate([action0, action1])


@dataclass
class ClothFoldPolicyConfig:
    approach_height: float = 0.02  # Very low approach - just above cloth
    grasp_height: float = 0.0      # At cloth level for contact
    lift_height: float = 0.02      # Minimal lift - stay within reach
    fold_height: float = 0.10
    retreat_height: float = 0.15   # Height to retreat to after release
    pos_gain: float = 1.0
    max_delta: float = 0.04
    grasp_dist_threshold: float = 0.05  # Increased threshold for easier transitions
    approach_steps: int = 15
    settle_steps: int = 10  # Wait time after reaching position before closing gripper
    grasp_steps: int = 25  # Time for gripper to close and vertices to attach
    lift_steps: int = 20
    fold_steps: int = 50  # More time for smooth cloth physics during fold
    release_steps: int = 8
    retreat_steps: int = 15  # Steps for retreat phase
    open_gripper: float = -1.0
    close_gripper: float = 1.0
    layout: str = "front-back"  # "front-back" or "parallel"
    # Grasp inward offset - target slightly inside corners for reachability
    grasp_inward_offset: float = 0.05  # Move 5cm toward cloth center (reduced for better vertex capture)


class ScriptedClothFoldPolicy:
    """Scripted fold: grasp corners and fold based on layout configuration.

    Supports two layouts:
    - "front-back": Robots at front/back (±y), grasp front corners, fold along Y-axis
    - "parallel": Robots on left side (-x), grasp left corners, fold along X-axis
    """

    def __init__(self, config: Optional[ClothFoldPolicyConfig] = None, debug: bool = False):
        self.config = config or ClothFoldPolicyConfig()
        self.phase = "approach"
        self.phase_step = 0
        self.debug = debug
        self._phase_step_counter = {}
        # Store initial grasp positions to avoid chasing moving cloth
        self._initial_grasp_pos0 = None
        self._initial_grasp_pos1 = None
        self._done = False

    @property
    def done(self) -> bool:
        """Returns True when the policy has completed all phases."""
        return self._done

    def reset(self):
        self.phase = "approach"
        self.phase_step = 0
        self._phase_step_counter = {}
        self._initial_grasp_pos0 = None
        self._initial_grasp_pos1 = None
        self._done = False

    def _pick_front_corners(self, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pick the two corners to grasp based on layout configuration.

        For "front-back" layout: Pick front corners (largest Y values)
        For "parallel" layout: Pick left corners (smallest X values)
        """
        if corners.shape != (4, 3):
            raise ValueError("cloth_corners must be shape (4, 3)")

        if self.config.layout == "parallel":
            # Parallel layout: Pick left-most corners (smallest X)
            # Robots are on left side (-x), grasp left corners
            xs = corners[:, 0]
            left_indices = np.argsort(xs)[:2]  # Two smallest X values
            left_corners = corners[left_indices]
            # Sort by Y to get consistent ordering (bottom, top)
            corner0 = left_corners[np.argmin(left_corners[:, 1])]  # Bottom-left
            corner1 = left_corners[np.argmax(left_corners[:, 1])]  # Top-left
            return corner0, corner1
        else:  # "front-back" layout (default)
            # Front-back layout: Pick front corners (largest Y)
            # Robots are at front (+y), grasp front corners
            ys = corners[:, 1]
            front_indices = np.argsort(ys)[-2:]  # Two largest Y values
            front = corners[front_indices]
            left = front[np.argmin(front[:, 0])]   # Front-left
            right = front[np.argmax(front[:, 0])]  # Front-right
            return left, right

    def _compute_arm_action(self, current: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        """Compute normalized action for OSC controller.

        The OSC controller expects normalized inputs [-1, 1] which it scales to output_max (0.05m).
        """
        delta = self.config.pos_gain * (target - current)
        # Normalize delta to [-1, 1] range for OSC controller
        osc_output_max = 0.05
        normalized_delta = delta / osc_output_max
        normalized_delta = np.clip(normalized_delta, -1.0, 1.0)
        return np.concatenate([normalized_delta, np.zeros(3), np.array([gripper])])

    def predict(self, obs: Dict) -> np.ndarray:
        eef0 = obs.get("robot0_eef_pos")
        eef1 = obs.get("robot1_eef_pos")
        corners = obs.get("cloth_corners")
        if eef0 is None or eef1 is None:
            raise ValueError("Observation is missing robot eef positions")
        if corners is None:
            raise ValueError("Observation is missing cloth_corners")

        corners = np.asarray(corners).reshape(4, 3)

        # Validate corners are reasonable
        if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
            raise ValueError(f"Invalid cloth corners detected (NaN/Inf): {corners}")

        left_corner, right_corner = self._pick_front_corners(corners)

        # Calculate grasp positions only on first call or during approach
        # After grasp, use stored positions to avoid chasing moving cloth
        if self._initial_grasp_pos0 is None or self.phase == "approach":
            # Compute cloth center for inward offset calculation
            cloth_center = corners.mean(axis=0)

            # Calculate inward-offset grasp positions (toward cloth center for reachability)
            inward = self.config.grasp_inward_offset
            if inward > 0:
                # Direction from corner toward center (XY only)
                dir0 = cloth_center[:2] - left_corner[:2]
                dir0 = dir0 / (np.linalg.norm(dir0) + 1e-6)
                dir1 = cloth_center[:2] - right_corner[:2]
                dir1 = dir1 / (np.linalg.norm(dir1) + 1e-6)

                grasp_pos0 = left_corner.copy()
                grasp_pos0[:2] += dir0 * inward
                grasp_pos1 = right_corner.copy()
                grasp_pos1[:2] += dir1 * inward
            else:
                grasp_pos0 = left_corner.copy()
                grasp_pos1 = right_corner.copy()

            # Store initial grasp positions when transitioning from approach
            if self.phase == "approach":
                self._initial_grasp_pos0 = grasp_pos0.copy()
                self._initial_grasp_pos1 = grasp_pos1.copy()
        else:
            # Use stored positions after grasp phase starts
            grasp_pos0 = self._initial_grasp_pos0
            grasp_pos1 = self._initial_grasp_pos1

        # Debug output
        if self.debug and self.phase_step % 10 == 0:
            print(f"[Cloth Fold Policy] Phase: {self.phase} | Step: {self.phase_step}")
            print(f"  Left corner: {left_corner}, Grasp pos: {grasp_pos0[:2]}")
            print(f"  Right corner: {right_corner}, Grasp pos: {grasp_pos1[:2]}")
            print(f"  Robot0 EEF: {eef0}")
            print(f"  Robot1 EEF: {eef1}")

        if self.phase == "approach":
            # Move above grasp positions (offset inward from corners)
            target0 = grasp_pos0.copy()
            target0[2] = left_corner[2] + self.config.approach_height
            target1 = grasp_pos1.copy()
            target1[2] = right_corner[2] + self.config.approach_height
            gripper = self.config.open_gripper

            dist0 = np.linalg.norm(target0 - eef0)
            dist1 = np.linalg.norm(target1 - eef1)

            # Transition to settle only when actually positioned correctly
            close_enough = (dist0 < self.config.grasp_dist_threshold and
                          dist1 < self.config.grasp_dist_threshold)

            if close_enough and self.phase_step >= self.config.approach_steps:
                self.phase = "settle"
                self.phase_step = 0

        elif self.phase == "settle":
            # Hold position at approach height with open grippers - let arms stabilize
            target0 = grasp_pos0.copy()
            target0[2] = left_corner[2] + self.config.approach_height
            target1 = grasp_pos1.copy()
            target1[2] = right_corner[2] + self.config.approach_height
            gripper = self.config.open_gripper

            if self.phase_step >= self.config.settle_steps:
                self.phase = "grasp"
                self.phase_step = 0

        elif self.phase == "grasp":
            # Grasp at inward-offset positions (within robot reach)
            target0 = grasp_pos0.copy()
            target0[2] = left_corner[2] + self.config.grasp_height
            target1 = grasp_pos1.copy()
            target1[2] = right_corner[2] + self.config.grasp_height
            gripper = self.config.close_gripper

            if self.phase_step >= self.config.grasp_steps:
                self.phase = "lift"
                self.phase_step = 0

        elif self.phase == "lift":
            # Lift from grasp positions
            target0 = grasp_pos0.copy()
            target0[2] = left_corner[2] + self.config.lift_height
            target1 = grasp_pos1.copy()
            target1[2] = right_corner[2] + self.config.lift_height
            gripper = self.config.close_gripper

            if self.phase_step >= self.config.lift_steps:
                self.phase = "fold"
                self.phase_step = 0

        elif self.phase == "fold":
            # Fold cloth by moving grippers to fixed target positions on the opposite side
            # Use fixed height at table surface (0.81m) to ensure proper fold
            table_z = 0.81
            if self.config.layout == "parallel":
                # Parallel layout: Move from left side (X=-0.45) toward center (X=0)
                # Target X is at center of cloth, keeping Y positions
                fold_x = 0.0  # Fold to center
                target0 = np.array([fold_x, grasp_pos0[1], table_z + 0.02])
                target1 = np.array([fold_x, grasp_pos1[1], table_z + 0.02])
            else:  # front-back layout
                # Front-back layout: Move from front (Y=+0.3) toward back (Y=-0.3)
                fold_y = -0.1  # Fold toward back
                target0 = np.array([grasp_pos0[0], fold_y, table_z + 0.02])
                target1 = np.array([grasp_pos1[0], fold_y, table_z + 0.02])
            gripper = self.config.close_gripper

            if self.phase_step >= self.config.fold_steps:
                self.phase = "release"
                self.phase_step = 0

        elif self.phase == "release":
            # Keep same position as fold phase but open grippers
            table_z = 0.81
            if self.config.layout == "parallel":
                fold_x = 0.0
                target0 = np.array([fold_x, grasp_pos0[1], table_z + 0.02])
                target1 = np.array([fold_x, grasp_pos1[1], table_z + 0.02])
            else:  # front-back layout
                fold_y = -0.1
                target0 = np.array([grasp_pos0[0], fold_y, table_z + 0.02])
                target1 = np.array([grasp_pos1[0], fold_y, table_z + 0.02])
            gripper = self.config.open_gripper

            if self.phase_step >= self.config.release_steps:
                self.phase = "retreat"
                self.phase_step = 0

        elif self.phase == "retreat":
            # Move grippers up and away from cloth
            table_z = 0.81
            if self.config.layout == "parallel":
                fold_x = 0.0
                target0 = np.array([fold_x, grasp_pos0[1], table_z + self.config.retreat_height])
                target1 = np.array([fold_x, grasp_pos1[1], table_z + self.config.retreat_height])
            else:  # front-back layout
                fold_y = -0.1
                target0 = np.array([grasp_pos0[0], fold_y, table_z + self.config.retreat_height])
                target1 = np.array([grasp_pos1[0], fold_y, table_z + self.config.retreat_height])
            gripper = self.config.open_gripper

            if self.phase_step >= self.config.retreat_steps:
                self.phase = "done"
                self.phase_step = 0
                self._done = True

        else:  # done or unknown phase
            # Hold position after completion
            table_z = 0.81
            if self.config.layout == "parallel":
                fold_x = 0.0
                target0 = np.array([fold_x, grasp_pos0[1], table_z + self.config.retreat_height])
                target1 = np.array([fold_x, grasp_pos1[1], table_z + self.config.retreat_height])
            else:  # front-back layout
                fold_y = -0.1
                target0 = np.array([grasp_pos0[0], fold_y, table_z + self.config.retreat_height])
                target1 = np.array([grasp_pos1[0], fold_y, table_z + self.config.retreat_height])
            gripper = self.config.open_gripper

        action0 = self._compute_arm_action(np.asarray(eef0), target0, gripper)
        action1 = self._compute_arm_action(np.asarray(eef1), target1, gripper)

        self.phase_step += 1

        return np.concatenate([action0, action1])
