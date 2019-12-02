# coding=utf-8
import numpy as np


def vec3(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def nvec3(x, y, z):
    v = vec3(x, y, z)
    return v / np.linalg.norm(v)


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
    mul = - z / z_ray
    if mul < 0:
        return None
    else:
        return ray_org + mul * ray, mul, z, z_ray


def collide_point2rect(px, py, ox, oy, w, h):
    return ox <= px <= ox + w and oy <= py <= oy + h


class CollideSurface:
    def __init__(self, org, vec_x, vec_y, nor, collide_back_face=False):
        self.org = org
        self.nor = nor
        self.vec_x = vec_x
        self.vec_y = vec_y
        self.lvec_x = np.linalg.norm(vec_x)
        self.lvec_y = np.linalg.norm(vec_y)
        self.nvec_x = vec_x / np.linalg.norm(vec_x)
        self.nvec_y = vec_y / np.linalg.norm(vec_y)
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
            return result[0]
        else:
            return None


if __name__ == '__main__':
    print(collide_ray2surface(vec3(100, 0, 100), nvec3(0, 0, 1), vec3(0, 100, 0), nvec3(0, -1, 0)))
