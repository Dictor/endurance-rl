# ready to run example: PythonClient/multirotor/hello_drone.py
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
    controlClient.takeoffAsync().join()

def moveForward():
    controlClient.moveByRollPitchYawThrottleAsync(1, 0, 0, 3).join()

def turnLeft():
    controlClient.rotateToYawAsync(-30).join()

def turnRight():
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
    return (bl, bc, br)