from control import tello_abs, tello_data
from world.world import *


def clamp_abs(v, minv, maxv):
    vv = max(min(abs(v), maxv), minv)
    return vv if v > 0 else (0 if v == 0 else -vv)


def go_abs(drone, state: tello_data.TelloData, dis):
    """
    预处理了yaw的移动，输入x, y, z基于原始坐标系就可以了
    """
    if state.mid == -1:
        raise BaseException("mid == -1")
    hpr = state.get_hpr()
    dis = np.matmul(hpr2matrix(hpr), dis)
    drone.go(dis[1], dis[0], dis[2])


def clamp_ang(ang):
    return (ang + 180) % 360 - 180


def look_at(backend: tello_abs.TelloBackendService, x, y, z, flag):
    while True:
        if not flag():
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            backend.drone.move_up(0.3)
        else:
            break
    state = backend.drone.get_state()
    dis = vec3(x, y, z) * 100 - vec3(state.x, state.y, state.z)
    ang = int(np.arctan2(dis[1], dis[0]) / np.pi * 180)
    dis_ang = state.mpry[1] - ang
    dis_ang = clamp_ang(dis_ang)
    if not flag():
        return
    if dis_ang < -1:
        backend.drone.rotate_cw(-dis_ang)
    elif dis_ang > 1:
        backend.drone.rotate_ccw(dis_ang)
    else:
        return


def goto(backend: tello_abs.TelloBackendService, x, y, z, flag, itridx=0, tol=0.45):
    if itridx >= 5:
        return
    while True:
        if not flag():
            return
        state = backend.drone.get_state()
        if state.mid == -1:
            backend.drone.move_up(0.3)
        else:
            break
    if not flag():
        return
    state = backend.drone.get_state()
    dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
    dis_f = min(150.0, np.linalg.norm(vec3(dis[0], dis[1], 0)))/100.0
    dis_u = dis[2]/100.0
    if dis_f <= 0.25 and abs(dis_u) <= 0.25:
        return

    look_at(backend, x, y, z, flag)
    state = backend.drone.get_state()
    dis = vec3(x, y, z) * 100 - vec3(state.x, state.y, state.z)
    dis_f = max(0.0, min(150.0, np.linalg.norm(vec3(dis[0], dis[1], 0)) - 10)) / 100.0
    dis_u = dis[2] / 100.0
    if not flag():
        return
    if dis_f > 0.25 or abs(dis_u) > 0.15:
        backend.drone.go(dis_f, 0, dis_u)
        state = backend.drone.get_state()
        if state.mid != -1:
            dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
            dis[2] *= 1.5
            dis = np.linalg.norm(dis/100.0)
            print("dis %.2f" % dis)
            if dis > tol:
                goto(backend, x, y, z, flag, itridx=itridx + 1, tol=tol)


if __name__ == '__main__':
    print(clamp_ang(100))
    print(clamp_ang(400))
    print(clamp_ang(180))
    print(clamp_ang(-30))
    print(clamp_ang(-190))

