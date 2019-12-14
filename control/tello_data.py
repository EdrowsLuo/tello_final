from world.world import *


class CameraModel:
    def __init__(self, pixel_x, pixel_y, angle_v):
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.pixel = np.array([self.pixel_x, self.pixel_y], dtype=np.float64)
        self.angle_v = angle_v
        self.dis = pixel_y / 2.0 / np.tan(angle_v / 2)
        self.angle_h = 2 * np.arctan2(pixel_x / 2.0, self.dis)

    def get_ray(self, pixel_x, pixel_y):
        r = np.array([pixel_x, pixel_y], dtype=np.float64) - self.pixel / 2
        return nm(vec3(r[0], self.dis, -r[1]) / 100.0)


predata = None


class TelloData:
    camera = CameraModel(960, 720, 45.0 / 180.0 * np.pi)
    cameraMatrix = hpr2matrix(vec3(0, -12, 0) / 180.0 * np.pi)

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
            elif 'bat' in item:
                self.bat = int(item.split(':')[-1])

        pos = self.get_pos()
        hpr = self.get_hpr()
        self.raw = [
            'pos: (%.2f, %.2f, %.2f)' % (pos[0], pos[1], pos[2]),
            'hpr: (%.2f, %.2f, %.2f)' % (hpr[0], hpr[1], hpr[2])
        ]
        for s in statestr:
            self.raw.append(s)

    def get_pos(self):
        # 85 -> 110
        # 125 -> 160
        #
        return vec3(self.y - 40, self.x - 160, self.z - 30) / 100.0

    def get_hpr(self):
        # TODO: 验证mpry和hpr的关系
        return vec3(-self.mpry[1], self.mpry[0], self.mpry[2]) / 180.0 * np.pi

    def get_ray(self, pixel_x, pixel_y):
        """
        根据当前tello姿态计算对应像素点对应射线检测数据
        """
        ray = TelloData.camera.get_ray(pixel_x, pixel_y)
        ray = np.matmul(TelloData.cameraMatrix, ray)
        ray = np.matmul(hpr2matrix(self.get_hpr()), ray)
        return nm(ray)

    def get_look_at(self):
        """
        返回无人机正面方向
        """
        return nm(np.matmul(hpr2matrix(self.get_hpr()), vec3(0, 1, 0)))

    def get_front(self):
        """
        无视pitch和roll的正面方向
        """
        return nm(np.matmul(hpr2matrix(vec3(self.get_hpr()[0], 0, 0)), vec3(0, 1, 0)))
