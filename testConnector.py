import airsimConnector

airsimConnector.connect()
airsimConnector.reset()

while True:
    i = input()
    if i == "t":
        airsimConnector.takeOff()
    elif i == "r":
        airsimConnector.turnRight()
    elif i == "l":
        airsimConnector.turnLeft()
    elif i == "f":
        airsimConnector.moveForward()
    elif i == "q":
        exit()
