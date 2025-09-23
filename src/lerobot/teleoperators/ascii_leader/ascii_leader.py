#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_ascii_leader import AsciiLeaderConfig

logger = logging.getLogger(__name__)


class AsciiLeader(Teleoperator):
    """
    ASCII-based leader teleoperator (clone of KochLeader behaviour with different naming).
    """

    config_class = AsciiLeaderConfig
    name = "ascii_leader"

    def __init__(self, config: AsciiLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "right_shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "right_shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "right_elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "right_wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "right_wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "right_gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
                "left_shoulder_pan": Motor(7, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "left_shoulder_lift": Motor(8, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "left_elbow_flex": Motor(9, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "left_wrist_flex": Motor(10, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "left_wrist_roll": Motor(11, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "left_gripper": Motor(12, "xl330-m077", MotorNormMode.RANGE_0_100),
                "waist_roll": Motor(13, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "waist_linear": Motor(14, "xl330-m288", MotorNormMode.RANGE_M100_100),
            },
            # self.bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # self.bus.write("Drive_Mode", "left_elbow_flex", DriveMode.INVERTED.value)
        self.bus.write("Drive_Mode", "left_wrist_flex", DriveMode.INVERTED.value)
        self.bus.write("Drive_Mode", "right_wrist_flex", DriveMode.INVERTED.value)
        drive_modes = {motor: 1 if motor in ["left_wrist_flex", "right_wrist_flex"] else 0 for motor in self.bus.motors}

        # self.bus.write("Drive_Mode", "elbow_flex", DriveMode.NON_INVERTED.value)
        # drive_modes = {motor: 0 for motor in self.bus.motors}

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = [] # ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor not in  ["left_gripper", "right_gripper"]:
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # Use 'position control current based' for grippers to be limited by the limit of the current.
        # This allows each gripper to act as a trigger and spring back to an open position.
        for g in ("left_gripper", "right_gripper"):
            if g in self.bus.motors:
                self.bus.write("Operating_Mode", g, OperatingMode.CURRENT_POSITION.value)
                # Enable torque for the gripper so it can be used as a physical trigger.
                self.bus.enable_torque(g)

        # Set grippers' goal pos in current position mode so that we can use them as triggers.
        if self.is_calibrated:
            for g in ("left_gripper", "right_gripper"):
                if g in self.bus.motors:
                    self.bus.write("Goal_Position", g, self.config.gripper_open_pos)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            if motor not in ["waist_roll", "waist_linear"]:
                # if motor.startswith("right_"):
                continue
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
