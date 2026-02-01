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

    def _compute_arm_action(
        self, current: np.ndarray, target: np.ndarray, gripper: float
    ) -> np.ndarray:
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
            if (
                dist0 < self.config.grasp_dist_threshold
                and dist1 < self.config.grasp_dist_threshold
            ):
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
    grasp_height: float = 0.0  # At cloth level for contact
    lift_height: float = (
        0.04  # Moderate lift - enough clearance but cloth stays on table
    )
    fold_height: float = 0.10
    retreat_height: float = 0.15  # Height to retreat to after release
    pos_gain: float = 1.0
    max_delta: float = 0.04
    grasp_dist_threshold: float = 0.05  # Increased threshold for easier transitions
    approach_steps: int = 15
    settle_steps: int = 10  # Wait time after reaching position before closing gripper
    grasp_steps: int = (
        20  # Time for gripper to close and vertices to attach (reduced from 25)
    )
    lift_steps: int = 20
    fold_steps: int = 40  # Time for smooth cloth physics during fold (reduced from 50)
    release_steps: int = 8
    retreat_steps: int = 12  # Steps for retreat phase (reduced from 15)
    open_gripper: float = -1.0
    close_gripper: float = 1.0
    layout: str = "front-back"  # "front-back" or "parallel"
    # Grasp inward offset - target slightly inside corners for reachability
    grasp_inward_offset: float = (
        0.02  # Move 2cm toward cloth center (close to true edge)
    )
    # Position convergence tolerance for early phase exit
    position_tolerance: float = 0.02  # 2cm - arms within this distance = converged
    # Minimum steps before position-based early exit is allowed
    min_settle_steps: int = 5
    min_grasp_steps: int = 15
    min_lift_steps: int = 10
    min_fold_steps: int = 20
    min_release_steps: int = 5
    min_retreat_steps: int = 5


class ScriptedClothFoldPolicy:
    """Scripted fold: grasp corners and fold based on layout configuration.

    Supports two layouts:
    - "front-back": Robots at front/back (±y), grasp front corners, fold along Y-axis
    - "parallel": Robots on left side (-x), grasp left corners, fold along X-axis
    """

    def __init__(
        self, config: Optional[ClothFoldPolicyConfig] = None, debug: bool = False
    ):
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
        self._fold_target0 = None
        self._fold_target1 = None
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
            left = front[np.argmin(front[:, 0])]  # Front-left
            right = front[np.argmax(front[:, 0])]  # Front-right
            return left, right

    def _compute_arm_action(
        self, current: np.ndarray, target: np.ndarray, gripper: float
    ) -> np.ndarray:
        """Compute normalized action for OSC controller.

        The OSC controller expects normalized inputs [-1, 1] which it scales to output_max (0.05m).
        """
        delta = self.config.pos_gain * (target - current)
        osc_output_max = 0.05
        normalized_delta = delta / osc_output_max
        normalized_delta = np.clip(normalized_delta, -1.0, 1.0)
        return np.concatenate([normalized_delta, np.zeros(3), np.array([gripper])])

    def _check_phase_complete(
        self,
        eef0: np.ndarray,
        target0: np.ndarray,
        eef1: np.ndarray,
        target1: np.ndarray,
        min_steps: int,
        max_steps: int,
        phase_name: str,
        allow_early_exit: bool = True,
    ) -> bool:
        dist0 = np.linalg.norm(target0 - eef0)
        dist1 = np.linalg.norm(target1 - eef1)
        converged = (
            dist0 < self.config.position_tolerance
            and dist1 < self.config.position_tolerance
        )

        max_reached = self.phase_step >= max_steps
        early_exit = allow_early_exit and converged and self.phase_step >= min_steps

        if self.debug and early_exit and not max_reached:
            print(
                f"  [Early exit] {phase_name} at step {self.phase_step} (dist0={dist0:.4f}, dist1={dist1:.4f})"
            )

        return max_reached or early_exit

    def _compute_fold_targets(
        self,
        grasp_pos0: np.ndarray,
        grasp_pos1: np.ndarray,
        corners: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fold target positions relative to cloth geometry.

        For parallel layout: Fold along X-axis toward cloth center
        For front-back layout: Fold along Y-axis toward cloth back
        """
        cloth_center = corners.mean(axis=0)

        # Use the actual grasped corner height + small offset (just above table)
        fold_z = grasp_pos0[2] + 0.02

        if self.config.layout == "parallel":
            # Parallel layout: Move from left side toward cloth center X
            xs = corners[:, 0]
            right_x = np.max(xs)
            left_x = np.min(xs)

            # Fold target: center of cloth along X
            fold_x = (left_x + right_x) / 2.0

            target0 = np.array([fold_x, grasp_pos0[1], fold_z])
            target1 = np.array([fold_x, grasp_pos1[1], fold_z])
        else:  # front-back layout
            # Front-back layout: Move from front toward back
            ys = corners[:, 1]
            front_y = np.max(ys)
            back_y = np.min(ys)

            # Fold target: toward back, ~30% from back edge
            fold_y = back_y + (front_y - back_y) * 0.3

            target0 = np.array([grasp_pos0[0], fold_y, fold_z])
            target1 = np.array([grasp_pos1[0], fold_y, fold_z])

        return target0, target1

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
            close_enough = (
                dist0 < self.config.grasp_dist_threshold
                and dist1 < self.config.grasp_dist_threshold
            )

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

            if self._check_phase_complete(
                eef0,
                target0,
                eef1,
                target1,
                self.config.min_settle_steps,
                self.config.settle_steps,
                "settle",
            ):
                self.phase = "grasp"
                self.phase_step = 0

        elif self.phase == "grasp":
            # Grasp at inward-offset positions (within robot reach)
            target0 = grasp_pos0.copy()
            target0[2] = left_corner[2] + self.config.grasp_height
            target1 = grasp_pos1.copy()
            target1[2] = right_corner[2] + self.config.grasp_height
            gripper = self.config.close_gripper

            if self._check_phase_complete(
                eef0,
                target0,
                eef1,
                target1,
                self.config.min_grasp_steps,
                self.config.grasp_steps,
                "grasp",
            ):
                self.phase = "lift"
                self.phase_step = 0

        elif self.phase == "lift":
            # Lift from grasp positions
            target0 = grasp_pos0.copy()
            target0[2] = left_corner[2] + self.config.lift_height
            target1 = grasp_pos1.copy()
            target1[2] = right_corner[2] + self.config.lift_height
            gripper = self.config.close_gripper

            if self._check_phase_complete(
                eef0,
                target0,
                eef1,
                target1,
                self.config.min_lift_steps,
                self.config.lift_steps,
                "lift",
            ):
                # Compute fold targets before transitioning to fold phase
                self._fold_target0, self._fold_target1 = self._compute_fold_targets(
                    grasp_pos0, grasp_pos1, corners
                )
                self.phase = "fold"
                self.phase_step = 0

        elif self.phase == "fold":
            # Fold cloth by moving grippers to computed target positions
            target0 = self._fold_target0.copy()
            target1 = self._fold_target1.copy()
            gripper = self.config.close_gripper

            if self._check_phase_complete(
                eef0,
                target0,
                eef1,
                target1,
                self.config.min_fold_steps,
                self.config.fold_steps,
                "fold",
                allow_early_exit=False,
            ):
                self.phase = "release"
                self.phase_step = 0

        elif self.phase == "release":
            # Keep same position as fold phase but open grippers
            target0 = self._fold_target0.copy()
            target1 = self._fold_target1.copy()
            gripper = self.config.open_gripper

            if self._check_phase_complete(
                eef0,
                target0,
                eef1,
                target1,
                self.config.min_release_steps,
                self.config.release_steps,
                "release",
                allow_early_exit=False,
            ):
                self.phase = "retreat"
                self.phase_step = 0

        elif self.phase == "retreat":
            # Move grippers up and away from cloth
            target0 = self._fold_target0.copy()
            target0[2] += self.config.retreat_height
            target1 = self._fold_target1.copy()
            target1[2] += self.config.retreat_height
            gripper = self.config.open_gripper

            if self._check_phase_complete(
                eef0,
                target0,
                eef1,
                target1,
                self.config.min_retreat_steps,
                self.config.retreat_steps,
                "retreat",
            ):
                self.phase = "done"
                self.phase_step = 0
                self._done = True

        else:  # done or unknown phase
            # Hold position after completion (use fold targets if available)
            if self._fold_target0 is not None and self._fold_target1 is not None:
                target0 = self._fold_target0.copy()
                target0[2] += self.config.retreat_height
                target1 = self._fold_target1.copy()
                target1[2] += self.config.retreat_height
            else:
                # Fallback for edge cases where fold targets weren't computed
                target0 = grasp_pos0.copy()
                target0[2] += self.config.retreat_height
                target1 = grasp_pos1.copy()
                target1[2] += self.config.retreat_height
            gripper = self.config.open_gripper

        action0 = self._compute_arm_action(np.asarray(eef0), target0, gripper)
        action1 = self._compute_arm_action(np.asarray(eef1), target1, gripper)

        self.phase_step += 1

        return np.concatenate([action0, action1])
