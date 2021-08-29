# ready to run example: PythonClient/multirotor/hello_drone.py
import math
import airsim
import os
import threading
import numpy as np
from PIL import Image

controlClient = airsim.MultirotorClient()
sensorClient = airsim.MultirotorClient()

def connect():
    controlClient.confirmConnection()
    controlClient.enableApiControl(True)
    controlClient.armDisarm(True)
    sensorClient.confirmConnection()
    sensorClient.enableApiControl(True)

def takeOff():
    print("tf")
    controlClient.takeoffAsync().join()

def reset():
    print("rst")
    moveOrigin()
    takeOff()

def moveOrigin():
    print("mo")
    position = airsim.Vector3r(0, 0, 0)
    heading = airsim.utils.to_quaternion(0, 0, 0)
    pose = airsim.Pose(position, heading)
    controlClient.simSetVehiclePose(pose, True)

def moveForward():
    print("mf")
    controlClient.moveByVelocityAsync(1, 0, 0, 1).join()

def turnLeft():
    print("ml")
    controlClient.rotateToYawAsync(-30).join()

def turnRight():
    print("mr")
    controlClient.rotateToYawAsync(30).join()

def getDistance():
    df = sensorClient.getDistanceSensorData(distance_sensor_name = "DistanceForward").distance
    db = sensorClient.getDistanceSensorData(distance_sensor_name = "DistanceBack").distance
    dl = sensorClient.getDistanceSensorData(distance_sensor_name = "DistanceLeft").distance
    dr = sensorClient.getDistanceSensorData(distance_sensor_name = "DistanceRight").distance
    return (df, db, dl, dr)

def getBright():
    ret = sensorClient.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])[0]
    img1d = np.fromstring(ret.image_data_uint8, dtype=np.uint8) #get numpy array
    smallImg = img1d.reshape(ret.height, ret.width, 3)
    r = smallImg[:,:,0]
    g = smallImg[:,:,1]
    b = smallImg[:,:,2]
    grayArr = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grayImg = Image.fromarray(grayArr)
    grayImg.thumbnail([3, 3])
    smallArr = np.array(grayImg)
    bl = smallArr[1, 0]
    bc = smallArr[1, 1]
    br = smallArr[1, 2]
    print("br : {0}".format((bl, bc, br)))
    return (bl, bc, br)

def getGoalDistance():
    vpos = sensorClient.simGetVehiclePose().position
    vpos = [vpos.x_val, vpos.y_val, vpos.z_val]
    gpos = [50, 0, 0]
    d =  math.sqrt(math.pow(vpos[0] - gpos[0], 2) + math.pow(vpos[1] - gpos[1], 2))
    print("gd : {0}".format(d))
    return d