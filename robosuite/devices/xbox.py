"""
Driver class for Keyboard controller.
"""

import time

import numpy as np
import pygame

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class XboxController(Device):
    """
    A minimalistic driver class for a Xbox controller.
    Args:

    remove -
        #pos_sensitivity (float): Magnitude of input position command scaling
        #rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, 
                 pos_sensitivity=1.0, 
                 rot_sensitivity=1.0
    ):
            
        pygame.init()
        pygame.joystick.init()


        if pygame.joystick.get_init():
            print("Joystick module IS initialized!")
        
        if pygame.joystick.get_init() == False:
            print("Joystick module not initialized!")
            return

        if pygame.joystick.get_count() == 0:
            print("No controllers found")
            return
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        
        self._display_controls()
        
        self.gripper_signal = False 

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False        
        

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("Start", "reset simulation")
        print_command("Left stick", "move arm in x-y plane")
        print_command("Right stick", "rotate arm")
        print_command("HOLD LEFT BUMPER + Right stick", "rotate gripper")
        print_command("Left trigger", "Close gripper")
        print_command("Right trigger", "Open gripper")
        print_command("Y", "Raise the arm")
        print_command("A", "Lower the arm")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._control = np.zeros(6)
        
        self.gripper_signal = False 

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )
    
    def _read_controller_state(self):

        """Listener method that keeps reading controller state."""   

        DEADZONE = 0.15  # Ignore small stick drift
        TRIGGER_THRESHOLD = 0.5
        
        pygame.event.pump()
        
        if not (self.joystick and self._enabled):
            return
        
        # Read all axes at once (like SpaceMouse reads all DOF from one message)
        left_x = self.joystick.get_axis(1)   # Left stick X
        left_y = self.joystick.get_axis(0)   # Left stick Y
        right_x = self.joystick.get_axis(3)  # Right stick X 
        right_y = self.joystick.get_axis(2)  # Right stick Y 
        lt = self.joystick.get_axis(4)       # Left trigger
        rt = self.joystick.get_axis(5)       # Right trigger
        

        # Debugging axis
        if abs(left_x) > DEADZONE :
            left_x = left_x 
            #print(f"left_x = {left_x}")
        else: 
            left_x = 0.0
        
        if abs(left_y) > DEADZONE :
            left_y = left_y
            #print(f"left_y : {left_y}")
        else:
            left_y =  0.0

        '''
        # Changing buttons to go from discrete --> continue.         
        button_y = self.joystick.get_button(3)  # Y button - raise arm
        button_a = self.joystick.get_button(0)  # A button - lower arm
        '''

        left_bumper = self.joystick.get_button(9)  # left_bumper (LB) - control yaw "mode"
        if left_bumper:
            if abs(right_x) > DEADZONE:
                self.z = -right_x
                #print(f"Right x : {right_x}")
                #print(f"Right y : {right_y}")
            else:
                self.z = 0
        else:
            if abs(right_x) > DEADZONE:
                right_x = right_x 
                #print(f"right_x : {right_x}")
            else: 
                right_x = 0.0
            
            if abs(right_y) > DEADZONE :
                right_y = right_y 
                #print(f"right_y : {right_y}")
            else:
                right_y = 0.0
        
        '''
        # Apply deadzone
        left_x = left_x if abs(left_x) > DEADZONE else 0.0
        left_y = left_y if abs(left_y) > DEADZONE else 0.0
        right_x = right_x if abs(right_x) > DEADZONE else 0.0
        right_y = right_y if abs(right_y) > DEADZONE else 0.0
        '''

        self.x = left_x
        self.y = left_y 

        right_bumper = self.joystick.get_button(10)
        if right_bumper:
            self.roll = 0
            self.pitch = 0
            self.yaw = right_y
        else:
            if not left_bumper:
                self.roll = right_x
                self.pitch = -right_y
                self.yaw = 0.0

        # Update control vector (all 6 DOF simultaneously like SpaceMouse)

        if abs(self.x) < DEADZONE and self.x > 0:
            print("x")
        
        if abs(self.y) < DEADZONE and self.y > 0:
            print("y")
        
        if abs(self.z) < DEADZONE and self.z > 0:
            print("z")

        self._control = [
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
        ]
        
        # Using RT as the main gripper button (matches SpaceMouse left button behavior)
        if rt > TRIGGER_THRESHOLD:
            self.gripper_signal = True
        elif lt > TRIGGER_THRESHOLD:
            self.gripper_signal = False
        else:
            # Released both - maintain last state
            pass
        
        # START button resets
        if self.joystick.get_button(6):  # START button
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()

    def run(self):
        self._read_controller_state()

    @property
    def control(self):
        """
        Grabs current pose of Xbox controller

        Returns:
            np.array: 6-DoF control value
        """
        self._read_controller_state()
        #print(self._control)
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we pressed the gripper button (RT/LT)
        """
        if self.gripper_signal:
            return 1.0
        return 0.0