from control import tello_abs, tello_data
from world.world import *
import sl4p

go_logger = sl4p.Sl4p('__go__')


def clamp_abs(v, minv, maxv):
    vv = max(min(abs(v), maxv), minv)
    return vv if v > 0 else (0 if v == 0 else -vv)


def find_most_possible_object(collect):
    poss = []
    for ss in collect:
        if collect[ss]['count'] < 6 or collect[ss]['max_conf'] < 0.7 or collect[ss]['object_conf'] < 0.2:
            continue
        else:
            poss.append(collect[ss])
    if len(poss) == 0:
        return None
    if len(poss) == 1:
        return poss[0]
    if len(poss) > 1:
        max_obj = None
        for s in poss:
            if max_obj is None or s['object_conf'] > max_obj['object_conf']:
                max_obj = s
        return max_obj


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
    if dis_f > 0.15 or abs(dis_u) > 0.15:
        if abs(dis_u) > 0.15:
            sign = -1 if dis_u < 0 else 1
            dis_u = sign * max(0.2, abs(dis_u))
        backend.drone.go(dis_f, 0, dis_u)
        state = backend.drone.get_state()
        if state.mid != -1:
            dis = vec3(x, y, z)*100 - vec3(state.x, state.y, state.z)
            dis[2] *= 2
            dis = np.linalg.norm(dis/100.0)
            go_logger.info("dis %.2f" % dis)
            if dis > tol:
                goto(backend, x, y, z, flag, itridx=itridx + 1, tol=tol)


if __name__ == '__main__':
    print(clamp_ang(100))
    print(clamp_ang(400))
    print(clamp_ang(180))
    print(clamp_ang(-30))
    print(clamp_ang(-190))

