# ready to run example: PythonClient/multirotor/hello_drone.py
import math
import airsim
import os
import threading
import numpy as np
from PIL import Image
import time


class airsimConnector():
    def __init__(self, port, name=""):
        self.controlLock = threading.Lock()
        self.controlClient = airsim.MultirotorClient(port=port)
        self.sensorClient = airsim.MultirotorClient(port=port)
        self.name = name
        self.yaw = 0

    def print(self, v):
        return
        print("[{0}] {1}".format(self.name, v))

    def connect(self):
        self.controlClient.confirmConnection()
        self.controlClient.enableApiControl(True)
        self.controlClient.armDisarm(True)
        self.sensorClient.confirmConnection()
        self.sensorClient.enableApiControl(True)
        self.sensorClient.takeoffAsync().join()

    def takeOff(self):
        self.print("take off")
        self.controlClient.takeoffAsync().join()

    def reset(self):
        self.print("reset")
        self.yaw = 0
        self.moveOrigin()

    def moveOrigin(self):
        self.controlLock.acquire()
        self.print("move origin")
        position = airsim.Vector3r(0, 0, -8)
        heading = airsim.utils.to_quaternion(0, 0, 0)
        pose = airsim.Pose(position, heading)
        self.controlClient.moveByVelocityAsync(0, 0, 0, 0.5).join()
        self.controlClient.simSetVehiclePose(pose, False)
        time.sleep(1)
        self.controlLock.release()

    def moveForward(self):
        self.controlLock.acquire()
        yaw = self.addYawAngle(0)
        self.print("move front, yaw={0}".format(yaw))
        yaw = (yaw / 360) * (2 * math.pi)
        vx = math.cos(yaw) * 1
        vy = math.sin(yaw) * 1
        self.controlClient.moveByVelocityAsync(vx, vy, 0, 1).join()
        self.controlLock.release()

    def turnLeft(self):
        self.controlLock.acquire()
        self.print("turn left")
        self.controlClient.rotateToYawAsync(self.addYawAngle(-15)).join()
        self.controlLock.release()

    def turnRight(self):
        self.controlLock.acquire()
        self.print("turn right")
        self.controlClient.rotateToYawAsync(self.addYawAngle(15)).join()
        self.controlLock.release()

    def getDistance(self):
        df = self.sensorClient.getDistanceSensorData(
            distance_sensor_name="DistanceForward").distance
        db = self.sensorClient.getDistanceSensorData(
            distance_sensor_name="DistanceBack").distance
        dl = self.sensorClient.getDistanceSensorData(
            distance_sensor_name="DistanceLeft").distance
        dr = self.sensorClient.getDistanceSensorData(
            distance_sensor_name="DistanceRight").distance
        return (df, db, dl, dr)

    def getBright(self):
        ret = self.sensorClient.simGetImages(
            [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])[0]
        img1d = np.fromstring(ret.image_data_uint8,
                              dtype=np.uint8)  # get numpy array
        smallImg = img1d.reshape(ret.height, ret.width, 3)
        r = smallImg[:, :, 0]
        g = smallImg[:, :, 1]
        b = smallImg[:, :, 2]
        grayArr = 0.2989 * r + 0.5870 * g + 0.1140 * b
        grayImg = Image.fromarray(grayArr)
        grayImg.thumbnail([3, 3])
        smallArr = np.array(grayImg)
        bl = smallArr[1, 0]
        bc = smallArr[1, 1]
        br = smallArr[1, 2]
        return (bl, bc, br)

    def getGoalDistance(self):
        vpos = self.sensorClient.simGetVehiclePose().position
        vpos = [vpos.x_val, vpos.y_val, vpos.z_val]
        self.print("vpos:{0}".format(vpos))
        gpos = [60, 30, 0]
        d = math.sqrt(math.pow(vpos[0] - gpos[0], 2) +
                      math.pow(vpos[1] - gpos[1], 2))
        return d

    def isCollided(self):
        c = self.sensorClient.simGetCollisionInfo()
        return c.has_collided

    def addYawAngle(self, v):
        self.yaw = self.yaw + v
        if self.yaw > 360:
            self.yaw = self.yaw - 360
        elif self.yaw < -360:
            self.yaw = self.yaw + 360
        return self.yaw
