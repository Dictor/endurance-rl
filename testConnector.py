from airsimConnector import airsimConnector

conn = airsimConnector(int(input("port=")), "eval")

while True:
    print("distance: ", conn.getDistance())
    print("bright: ", conn.getBright())
    print("goal distance: ", conn.getGoalDistance())
    i = input()
    if i == "t":
        conn.takeOff()
    elif i == "r":
        conn.turnRight()
    elif i == "l":
        conn.turnLeft()
    elif i == "f":
        conn.moveForward()
    elif i == "q":
        exit()
    elif i == "o":
        conn.moveOrigin()
