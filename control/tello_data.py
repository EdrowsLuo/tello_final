
class TelloData:
    def __init__(self, state):
        statestr = state.split(';')
        for item in statestr:
            if 'mid:' in item:
                mid = int(item.split(':')[-1])
                self.mid = mid
            elif 'x:' in item:
                x = int(item.split(':')[-1])
                self.x = x
            elif 'z:' in item:
                z = int(item.split(':')[-1])
                self.z = -z
            elif 'mpry:' in item:
                mpry = item.split(':')[-1]
                mpry = mpry.split(',')
                self.mpry = [int(mpry[0]), int(mpry[1]), int(mpry[2])]
            # y can be recognized as mpry, so put y first
            elif 'y:' in item:
                y = int(item.split(':')[-1])
                self.y = y
            elif 'pitch:' in item:
                pitch = int(item.split(':')[-1])
                self.pitch = pitch
            elif 'roll:' in item:
                roll = int(item.split(':')[-1])
                self.roll = roll
            elif 'yaw:' in item:
                yaw = int(item.split(':')[-1])
                self.yaw = yaw
