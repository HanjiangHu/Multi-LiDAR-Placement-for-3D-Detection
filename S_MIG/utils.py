import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from sample_script import BresenhamInt3D

def BresenhamVec3D(vec1: list, vec2: list):
    return BresenhamInt3D(vec1[0], vec1[1], vec1[2], vec2[0], vec2[1], vec2[2])

def get_points(origin, distance, angle, min_clip, max_clip):
    point = get_point_at_distance_and_angle(*origin, distance, angle)
    point = np.clip(point, min_clip, max_clip)
    return np.array(BresenhamInt3D(origin[0], origin[1], 0, point[0], point[1], 0))


def get_3d_points(origin, distance, theta, phi, min_clip, max_clip):
    point = get_3d_point_at_distance_and_angle(*origin, distance, theta, phi)

    point = np.clip(point, min_clip, max_clip)
    return np.array(
        BresenhamInt3D(origin[0], origin[1], origin[2], point[0], point[1], point[2])
    )


def transform(world, lidar_origin):
    rotation_matrix = R.from_euler("zyx", lidar_origin[3:])

    if isinstance(world[0], list) or isinstance(world[0], np.ndarray):
        v_3f = np.array([[w[0], w[1], w[2], 1] for w in world])
    else:
        v_3f = np.array([world[0], world[1], world[2], 1])

    t = [[lidar_origin[0], lidar_origin[1], lidar_origin[2]]]

    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix.as_matrix()
    transformation_matrix[:3, 3] = np.array(t)

    cube_local = np.dot(np.linalg.inv(transformation_matrix), v_3f.T)
    return cube_local


def distance_something(world_coords, lidar_origin, beam_angle):
    local = transform(world_coords, lidar_origin).T
    local[:, 1] - local[:, 0]
    distance = local[:, 2] - math.tan(np.pi * (beam_angle) / 180) * np.sqrt(
        pow(local[:, 0], 2) + pow(local[:, 1], 2)
    )
    return abs(distance)


def get_points_in_proximity(world_coords, lidar_origin, beam_angle, beta):
    local = transform(world_coords, lidar_origin).T
    local[:, 1] - local[:, 0]
    distance = local[:, 2] - math.tan(np.pi * (beam_angle) / 180) * np.sqrt(
        pow(local[:, 0], 2) + pow(local[:, 1], 2)
    )
    distance = abs(distance)
    inds = distance < beta
    return world_coords[inds, :]


def get_all_points(shell_points, lidar_origin, beam_angles, beta):
    for beam_angle in beam_angles:
        for y in range(0, 314):
            y /= 100
            lidar_origin[5] = y
            d = get_points_in_proximity(shell_points, lidar_origin, beam_angle, beta)

            for point in d:
                all_points.extend(
                    BresenhamInt3D(
                        lidar_origin[0],
                        lidar_origin[1],
                        lidar_origin[2],
                        point[0],
                        point[1],
                        point[2],
                    )
                )

    all_points = np.array(all_points)
    all_points = np.unique(all_points, axis=0)
    return all_points


def get_shell_points():
    shell_points = []
    for i in range(40):
        for j in range(4):
            shell_points.append([60, i, j])
            shell_points.append([0, i, j])

    for i in range(60):
        for j in range(4):
            shell_points.append([i, 40, j])
            shell_points.append([i, 0, j])

    for i in range(60):
        for j in range(40):
            shell_points.append([i, j, 4])
            shell_points.append([i, j, 0])
    return shell_points


def get_point_at_distance_and_angle(origin_x, origin_y, distance, angle):
    angle = (angle / 180) * math.pi
    x = origin_x + (math.cos(angle) * distance)
    y = origin_y + (math.sin(angle) * distance)
    return round(x, 2), round(y, 2)


def get_3d_point_at_distance_and_angle(
    origin_x, origin_y, origin_z, distance, theta, phi
):
    theta = (theta / 180) * np.pi
    phi = (phi / 180) * np.pi
    x = distance * math.sin(theta) * math.cos(phi) + origin_x
    y = distance * math.sin(theta) * math.sin(phi) + origin_y
    z = distance * math.cos(theta) + origin_z
    return round(x), round(y), round(z)


def get_points_brute_force(
    cube_x_num,
    cube_y_num,
    cube_z_num,
    cube_resolution_x,
    cube_resolution_y,
    cube_resolution_z,
    dead_x_low,
    dead_x_high,
    dead_y_low,
    dead_y_high,
    dead_z_low,
    dead_z_high,
    lidar_origin,
    beam_angle,
    laser_num,
):
    for i in range(cube_x_num):
        for j in range(cube_y_num):
            for k in range(cube_z_num):
                cube_x_world = cube_resolution_x * (i + 0.5)
                cube_y_world = cube_resolution_y * (j + 0.5)
                cube_z_world = cube_resolution_z * (k + 0.5)
                if (
                    (cube_x_world >= dead_x_low and cube_x_world <= dead_x_high)
                    and (cube_y_world >= dead_y_low and cube_y_world <= dead_y_high)
                    and (cube_z_world >= dead_z_low and cube_z_world <= dead_z_high)
                ):
                    dead.add(f"{cube_x_world};{cube_y_world};{cube_z_world}")
                    #                 print(cube_x_world, cube_y_world, cube_z_world)
                    continue
                #             print(i,j,k)
                m = 0
                laser_index = 0

                for yaw in range(0, 31):
                    # R P Y
                    roll = lidar_origin[6 * m + 3]
                    pitch = lidar_origin[6 * m + 4]
                    yaw /= 10

                    rotation_matrix = transforms3d.euler.euler2mat(
                        roll, pitch, yaw
                    )  # [yaw, pitch, roll])

                    v_3f = [cube_x_world, cube_y_world, cube_z_world]

                    t = [
                        [
                            lidar_origin[6 * m],
                            lidar_origin[6 * m + 1],
                            lidar_origin[6 * m + 2],
                        ]
                    ]

                    transformation_matrix = np.identity(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = np.array(t)

                    cube_local = np.dot(
                        np.linalg.inv(transformation_matrix), [*v_3f, 1]
                    )

                    cube_x_local = cube_local[0]
                    cube_y_local = cube_local[1]
                    cube_z_local = cube_local[2]

                    for laser_index in range(laser_num):
                        if (
                            abs(
                                cube_z_local
                                - math.tan(np.pi * (beam_angle[laser_index]) / 180)
                                * np.sqrt(pow(cube_x_local, 2) + pow(cube_y_local, 2))
                            )
                            < beta
                        ):
                            something.append([cube_x_world, cube_y_world, cube_z_world])


    something = np.array(something)
    something = np.unique(something, axis=0)
    return something



def create_shell_face(num_1_cubes, num_2_cubes, num_3_cubes):
    shell_points = np.mgrid[0:num_2_cubes:1, 0:num_3_cubes:1].reshape(2, -1).T
    x_locs = np.ones(shell_points.shape[0], dtype=int) * num_1_cubes
    zero_locs = np.zeros(shell_points.shape[0], dtype=int) * 0

    x_locs = np.hstack((shell_points, x_locs.reshape(-1, 1)))
    zero_locs = np.hstack((shell_points, zero_locs.reshape(-1, 1)))

    shell_points = np.vstack((x_locs, zero_locs))
    shell_points = shell_points.astype(int)
    return shell_points


def create_cube_surface(nums):
    surface_x = create_shell_face(nums[0], nums[1], nums[2])
    surface_y = create_shell_face(nums[1], nums[2], nums[0])
    surface_z = create_shell_face(nums[2], nums[0], nums[1])
    return np.vstack((surface_x[:, [2, 0, 1]], surface_y[:, [1, 2, 0]], surface_z))


def get_normal_vectors():
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )


def get_points(num_x_cubes, num_y_cubes, num_z_cubes):
    return np.array(
        [
            [num_x_cubes / 2, -num_y_cubes / 2, -num_z_cubes / 2],
            [-num_x_cubes / 2, -num_y_cubes / 2, -num_z_cubes / 2],
            [num_x_cubes / 2, num_y_cubes / 2, -num_z_cubes / 2],
            [num_x_cubes / 2, -num_y_cubes / 2, -num_z_cubes / 2],
            [num_x_cubes / 2, num_y_cubes / 2, num_z_cubes / 2],
            [-num_x_cubes / 2, -num_y_cubes / 2, -num_z_cubes / 2],
        ]
    )


def get_direction_vector(yaw, beam_angle):
    theta = np.pi * yaw / 180
    phi = np.pi * (90 - beam_angle) / 180
    x = 1 * np.cos(theta) * np.sin(phi)
    y = 1 * np.sin(theta) * np.sin(phi)
    z = 1 * np.cos(phi)

    return np.array([x, y, z])


def get_intersection(planePoint, planeNormal, linePoint, lineDirection):
    if np.dot(planeNormal, lineDirection) == 0:
        return None

    t = (np.dot(planeNormal, planePoint) - np.dot(planeNormal, linePoint)) / np.dot(
        planeNormal, lineDirection
    )
    return linePoint + t * lineDirection

def get_all_points_in_bbox(bbox: np.ndarray, scale: int):
    bbox_x = ((bbox[0,:]+30) * int(scale)).astype(int).flatten()#[:3]
    bbox_y = ((bbox[1,:]+20) * int(scale)).astype(int).flatten()#[:3]
    if np.all(bbox[2,:] >= 0):
        bbox_z = ((bbox[2, :]) * int(scale)).astype(int).flatten()
    else:
        bbox_z = ((bbox[2, :] - np.min(bbox[2, :])) * int(scale)).astype(int).flatten()  # [:3]
    """
       A -------- C
       /|      /|
      / |     / |
     /  |    /  |
    /  E|__ / __| G
   B-------- D  / 
    |   /  |   /  
    |  /   |  /
    | /    | /
    |/_____|/
    F      H

    A: 0
    B: 1
    C: 2
    D: 3
    E: 4
    F: 5
    G: 6
    H: 7
    """

    AB = BresenhamInt3D(bbox_x[0], bbox_y[0], bbox_z[0], bbox_x[1], bbox_y[1], bbox_z[1])
    AC = BresenhamInt3D(bbox_x[0], bbox_y[0], bbox_z[0], bbox_x[2], bbox_y[2], bbox_z[2])
    AE = BresenhamInt3D(bbox_x[0], bbox_y[0], bbox_z[0], bbox_x[4], bbox_y[4], bbox_z[4])
    BD = BresenhamInt3D(bbox_x[1], bbox_y[1], bbox_z[1], bbox_x[3], bbox_y[3], bbox_z[3])
    # BF = BresenhamInt3D(bbox_x[1], bbox_y[1], bbox_z[1], bbox_x[5], bbox_y[5], bbox_z[5])
    CD = BresenhamInt3D(bbox_x[2], bbox_y[2], bbox_z[2], bbox_x[3], bbox_y[3], bbox_z[3])
    CG = BresenhamInt3D(bbox_x[2], bbox_y[2], bbox_z[2], bbox_x[6], bbox_y[6], bbox_z[6])
    EG = BresenhamInt3D(bbox_x[4], bbox_y[4], bbox_z[4], bbox_x[6], bbox_y[6], bbox_z[6])
    EF = BresenhamInt3D(bbox_x[4], bbox_y[4], bbox_z[4], bbox_x[5], bbox_y[5], bbox_z[5])
    FH = BresenhamInt3D(bbox_x[5], bbox_y[5], bbox_z[5], bbox_x[7], bbox_y[7], bbox_z[7])
    GH = BresenhamInt3D(bbox_x[6], bbox_y[6], bbox_z[6], bbox_x[7], bbox_y[7], bbox_z[7])
    DH = BresenhamInt3D(bbox_x[3], bbox_y[3], bbox_z[3], bbox_x[7], bbox_y[7], bbox_z[7])

    # print(AE)
    BF = []
    for i in range(len(AE)):
        BF.append((AE[i][0]+bbox_x[1]-bbox_x[0], AE[i][1], AE[i][2]))

    assert len(AE) == len(BF), f"Length of segments dont match {len(AE)} != {len(BF)}"

    AEBF = []
    for i in range(len(AE)):
        AEBF.extend(BresenhamVec3D(AE[i], BF[i]))

    CDHG = []
    for i in range(len(AEBF)):
        CDHG.append((AEBF[i][0], AEBF[i][1]+bbox_y[3]-bbox_y[1], AEBF[i][2]))

    ABDCGHFE = []

    assert len(AEBF) == len(CDHG)


    for i in range(len(AEBF)):
        ABDCGHFE.extend(BresenhamVec3D(AEBF[i], CDHG[i]))

    return ABDCGHFE
