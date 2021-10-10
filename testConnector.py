from airsimConnector import airsimConnector

conn = airsimConnector(int(input("port=")), "eval")
conn.connect()

while True:
    print("distance: ", conn.getDistance())
    print("bright: ", conn.getBright())
    print("goal distance: ", conn.getGoalDistance())
    pos = conn.getPosition()
    print("position: {:.2f}, {:.2f}, {:.2f}".format(
        pos.x_val, pos.y_val, pos.z_val))
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
    elif i == "m":
        conn.moveTo(int(input("x=")), int(input("y=")), int(input("z=")))
    else:
        print("unknown command!")
