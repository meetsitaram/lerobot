import cv2
import numpy as np
import torch

from lerobot.common.kinematics import KochKinematics
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.koch import KochRobot
from lerobot.common.robot_devices.utils import busy_wait

GRIPPER_TIP_Z_BOUNDS = (0.004, 0.1)
GRIPPER_TIP_X_BOUNDS = (-0.15, 0.15)
GRIPPER_TIP_Y_BOUNDS = (-0.25, -0.05)
GRIPPER_TIP_BOUNDS = [GRIPPER_TIP_X_BOUNDS, GRIPPER_TIP_Y_BOUNDS, GRIPPER_TIP_Z_BOUNDS]


def is_in_bounds(gripper_tip_pos, buffer=0):
    for i, bounds in enumerate(GRIPPER_TIP_BOUNDS):
        assert (bounds[1] - bounds[0]) / 2 > buffer
        if gripper_tip_pos[i] < bounds[0] + buffer or gripper_tip_pos[i] > bounds[1] - buffer:
            return False
    return True


def calc_smoothness_reward(
    action: np.ndarray,
    prior_action: np.ndarray | None = None,
    first_order_coeff: float = -1.0,
    second_order_coeff: float = -1.0,
):
    reward = first_order_coeff * np.linalg.norm(action)
    if prior_action is not None:
        reward += second_order_coeff * np.linalg.norm(action - prior_action)
    return reward


def calc_reward_cube_push(
    obj_mask,
    goal_mask,
    action,
    gripper_tip_pos,
    distance_reward_coeff: float = 1 / 45,
    action_magnitude_reward_coeff: float = -1 / 25,
) -> tuple[float, bool]:
    intersection_area = np.count_nonzero(np.bitwise_and(obj_mask, goal_mask))

    success = False
    if intersection_area <= 0:
        # Find the minimum distance between the object and the goal.
        goal_contour = cv2.findContours(
            goal_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )[0]
        obj_contour = cv2.findContours(
            obj_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )[0]
        obj_points = np.vstack(obj_contour).squeeze()  # shape (N, 2)
        goal_points = np.vstack(goal_contour).squeeze()  # shape (M, 2)
        distances = np.linalg.norm(obj_points[:, None] - goal_points[None, :], axis=-1)  # shape (N, M, 2)
        reward = -np.min(distances) * distance_reward_coeff
    elif intersection_area > 0:
        reward = intersection_area / np.count_nonzero(obj_mask)
        success = reward == 1

    reward += action_magnitude_reward_coeff * np.abs(action).max()

    do_terminate = False

    if not is_in_bounds(gripper_tip_pos):
        do_terminate = True
        reward -= 1

    if success:
        do_terminate = True
        reward += 1

    # Lose 1 for each step to encourage faster completion.
    reward -= 1

    return reward, success, do_terminate


def calc_reward_joint_goal(
    current_joint_pos,
    action: np.ndarray | None = None,
    prior_action: np.ndarray | None = None,
    first_order_smoothness_coeff: float = -1.0,
    second_order_smoothness_coeff: float = -1.0,
):
    # Whole arm
    goal = np.array([87, 82, 91, 65, 3, 30])
    curr = current_joint_pos
    reward = -np.abs(goal - curr).mean() / 10
    success = np.abs(goal - curr).max() <= 3

    do_terminate = False

    gripper_tip_pos = KochKinematics.fk_gripper_tip(current_joint_pos)[:3, -1]
    if not is_in_bounds(gripper_tip_pos):
        reward -= 1
        do_terminate = True

    if success:
        do_terminate = True
        reward += 1

    if action is not None:
        reward += calc_smoothness_reward(
            action, prior_action, first_order_smoothness_coeff, second_order_smoothness_coeff
        )

    return reward, success, do_terminate


def reset_for_joint_pos(robot: KochRobot):
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    reset_pos = robot.follower_arms["main"].read("Present_Position")
    while True:
        # For reference: goal = [87, 82, 91, 65, 3, 30]
        reset_pos[0] = np.random.uniform(70, 110)
        reset_pos[1] = np.random.uniform(70, 90)
        reset_pos[2] = np.random.uniform(80, 110)
        reset_pos[3] = np.random.uniform(45, 90)
        reset_pos[4] = np.random.uniform(-50, 50)
        reset_pos[5] = np.random.uniform(0, 90)
        if is_in_bounds(KochKinematics.fk_gripper_tip(reset_pos)[:3, -1], buffer=0.02):
            break
    reset_pos = torch.from_numpy(reset_pos)
    while True:
        robot.send_action(reset_pos)
        current_pos = robot.follower_arms["main"].read("Present_Position")
        busy_wait(1 / 30)
        if np.all(np.abs(current_pos - reset_pos.numpy())[-3:] < np.array([10, 3, 3])):
            break
