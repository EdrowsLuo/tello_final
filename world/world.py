# coding=utf-8
import numpy as np


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def nvec3(x, y, z):
    v = vec3(x, y, z)
    return v/np.linalg.norm(v)


def nm(v):
    return v/np.linalg.norm(v)


def vec2hpr(v):
    return vec3(np.arctan2(-v[0], v[1]), np.arctan2(v[2], np.linalg.norm(v[0:-1])), 0)


def hpr2vec(hpr):
    return nvec3(np.tan(-hpr[0]), 1, np.tan(hpr[1]) / np.cos(hpr[0]))


def hpr_add(v1, v2):
    pass


def hpr2matrix(t):
    ca = np.cos(t[2])
    sa = np.sin(t[2])
    cb = np.cos(t[1])
    sb = np.sin(t[1])
    cg = np.cos(t[0])
    sg = np.sin(t[0])
    return np.array([
        [ca*cg-cb*sa*sg, -cb*cg*sa-ca*sg, sa*sb],
        [cg*sa+ca*cb*sg, ca*cb*cg-sa*sg, -ca*sb],
        [sb*sg, cg*sb, cb]
    ], dtype=np.float64)


def detect_parallel(ray, nor, tol=0.1):
    return abs(np.dot(ray, nor)) < tol


def collide_ray2surface(ray_org, ray, org, nor, tol=0.1):
    if detect_parallel(ray, nor):
        # 如果平行则无碰撞
        return None
    z = np.dot(ray_org - org, nor)
    z_ray = np.dot(ray, nor)
    if abs(z_ray) < tol:
        return None
    mul = - z/z_ray
    if mul < 0:
        return None
    else:
        return ray_org + mul*ray, mul, z, z_ray


def collide_point2rect(px, py, ox, oy, w, h):
    return ox <= px <= ox + w and oy <= py <= oy + h


def world_model():
    list = [
        {
            'tag': ['box'],
            'id': 'box1',
            'model': CollideBox(org=vec3(-.395, 0.52, 0), size=vec3(.395, 1.20, 1.06))
        },
        {
            'tag': ['box'],
            'id': 'box2',
            'model': CollideBox(org=vec3(.395 + .98 + .395 + 1.0, 1.3, 0), size=vec3(.395, 1.20, 1.46))
        },
        {
            'tag': ['box'],
            'id': 'box3',
            'model': CollideBox(org=vec3(.395 + .98, 1.8, 0), size=vec3(.395, 1.20, 1.86))
        },
        {
            'tag': ['box'],
            'id': 'box4',
            'model': CollideBox(org=vec3(.395 + .45, 1.8 + 1.2 + 1.23, 0), size=vec3(1.20, 0.395, 1.66))
        },
        {
            'tag': ['box'],
            'id': 'box5',
            'model': CollideBox(org=vec3(-.395, .52 + 1.2 + 1.8, 0), size=vec3(.395, 1.20, 1.26))
        }
    ]

    itembox = []
    for m in list:
        size = vec3(0.40, 0.40, 0.40)
        raw = m['model']  # type: CollideBox
        itembox.append({
            'tag': ['box', 'item'],
            'id': 'item' + m['id'],
            'model': CollideBox(org=raw.org + raw.size / 2.0 - size / 2.0 + vec3(0, 0, raw.size[2] / 2.0 + size[2] / 2.0), size=size),
            'model_big': CollideBox(org=raw.org + raw.size/2.0 - size/2.0 + vec3(0, 0, raw.size[2]/2.0 + size[2]/2.0) - 0.1,
                                size=size + 0.2),
        })

    for m in itembox:
        list.append(m)

    l = 0
    index = 0
    sy = 0.03
    for i in range(3):
        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 0), size=vec3(0.06, sy, 2.06))
            }
        )
        l += 0.06
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 0), size=vec3(0.42, sy, 1.26))
            }
        )
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 1.26 + 0.3), size=vec3(0.42, sy, 0.1))
            }
        )
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 1.26 + 0.3 + 0.1 + 0.3),
                                    size=vec3(0.42, sy, 2.06 - (1.26 + 0.3 + 0.1 + 0.3)))
            }
        )
        l += 0.42
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 0), size=vec3(0.04, sy, 2.06))
            }
        )
        l += 0.04
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 0), size=vec3(0.42, sy, 1.26))
            }
        )
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 1.26 + 0.3), size=vec3(0.42, sy, 0.1))
            }
        )
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 1.26 + 0.3 + 0.1 + 0.3),
                                    size=vec3(0.42, sy, 2.06 - (1.26 + 0.3 + 0.1 + 0.3)))
            }
        )
        l += 0.42
        index += 1

        list.append(
            {
                'tag': ['box', 'wall'],
                'id': 'wall%d'%index,
                'model': CollideBox(org=vec3(l, 0, 0), size=vec3(0.06, sy, 2.06))
            }
        )
        l += 0.06
        index += 1
    list.append(
        {
            'tag': ['surface', 'wall'],
            'id': 'wall_surface,',
            'model': CollideSurface(org=vec3(0, 0, 0), vec_x=vec3(3, 0, 0), vec_y=vec3(0, 0, 2.06), nor=vec3(0, -1, 0)),
            'model_box': CollideBox(org=vec3(0, 0.01, 0), size=[3, 0.01, 2.06])
        }
    )
    return list


class CollideSurface:
    def __init__(self, org, vec_x, vec_y, nor, collide_back_face=False):
        self.org = org
        self.nor = nor
        self.vec_x = vec_x
        self.vec_y = vec_y
        self.lvec_x = np.linalg.norm(vec_x)
        self.lvec_y = np.linalg.norm(vec_y)
        self.nvec_x = vec_x/np.linalg.norm(vec_x)
        self.nvec_y = vec_y/np.linalg.norm(vec_y)
        self.collide_back_face = collide_back_face

    def collide_ray(self, ray_org, ray):
        result = collide_ray2surface(ray_org, ray, self.org, self.nor)
        if result is None:
            return None
        if not self.collide_back_face and result[2] <= 0:
            return None
        surface_vec = result[0] - self.org
        side_x = np.dot(surface_vec, self.nvec_x)
        side_y = np.dot(surface_vec, self.nvec_y)
        if collide_point2rect(side_x, side_y, 0, 0, self.lvec_x, self.lvec_y):
            return result[0], result[1]  # pos, mul
        else:
            return None


class CollideBox:
    def __init__(self, org, size):
        self.org = org
        self.size = size
        self.vec_x = vec3(size[0], 0, 0)
        self.vec_y = vec3(0, size[0], 0)
        self.vec_z = vec3(0, 0, size[0])
        self.surfaces = [
            CollideSurface(org, self.vec_x, self.vec_y, -nm(self.vec_z)),
            CollideSurface(org, self.vec_y, self.vec_z, -nm(self.vec_x)),
            CollideSurface(org, self.vec_z, self.vec_x, -nm(self.vec_y)),
            CollideSurface(org + self.vec_z, self.vec_x, self.vec_y, nm(self.vec_z)),
            CollideSurface(org + self.vec_x, self.vec_y, self.vec_z, nm(self.vec_x)),
            CollideSurface(org + self.vec_y, self.vec_z, self.vec_x, nm(self.vec_y))
        ]  # type: list[CollideSurface]

    def collide_ray(self, ray_org, ray):
        for i in range(len(self.surfaces)):
            result = self.surfaces[i].collide_ray(ray_org, ray)
            if result is not None:
                return result[0], result[1], i
        return None


if __name__ == '__main__':
    print(collide_ray2surface(vec3(100, 0, 100), nvec3(0, 0, 1), vec3(0, 100, 0), nvec3(0, -1, 0)))
