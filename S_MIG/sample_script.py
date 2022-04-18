#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import toml
# import transforms3d

from scipy.spatial.transform import Rotation as R


def BresenhamInt3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Y-axis"
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints


# In[3]:


def main():
    (x1, y1, z1) = (7, 3, 9)
    (x2, y2, z2) = (20, 26, 4)
    p = np.array(BresenhamInt3D(x1, y1, z1, x2, y2, z2)).astype(float)
    params = toml.load("multihyper.toml")
    x = [30, 20,  2,  0.0,    -0.0]
    points = get_points_covered_by_lidar_config(
        params["pom"], params["lidar"], x, 1
    )



def transform(world, lidar_origin):
    rotation_matrix = R.from_euler("ZYX", lidar_origin[3:])

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
    hyp = np.sqrt(pow(local[:, 0], 2) + pow(local[:, 1], 2))

    distance = local[:, 2] - math.tan(np.pi * (beam_angle) / 180) * hyp
    distance = abs(distance)
    inds = distance < beta
    return world_coords[inds, :]


def remove_points_from_dead_zone(points, dead_zones):
    x = np.logical_and(points[:, 0] > dead_zone[0], points[:, 0] < dead_zone[1])
    y = np.logical_and(points[:, 1] > dead_zone[2], points[:, 1] < dead_zone[3])
    z = np.logical_and(points[:, 2] > dead_zone[4], points[:, 2] < dead_zone[5])
    return points[np.logical_not(x | y | x)]


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
    #     surface_y = np.moveaxis(surface_y, [0,1,2], [1,2,0])
    surface_z = create_shell_face(nums[2], nums[0], nums[1])
    #     surface_z = np.moveaxis(surface_z, [0,1,2], [2,0,1])
    # print(surface_z.shape, surface_x.shape, surface_y.shape)
    return np.vstack((surface_x[:, [2, 0, 1]], surface_y[:, [1, 2, 0]], surface_z))


def get_normal_vectors():
    return np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )


def get_points(num_x_cubes, num_y_cubes, num_z_cubes):
    return np.array(
        [
            [num_x_cubes / 2,0, 0],
            [-num_x_cubes / 2, 0, 0],
            [0, num_y_cubes / 2, 0],
            [0, -num_y_cubes / 2, 0],
            [0, 0, num_z_cubes / 2],
            [0, 0, -num_z_cubes / 2],
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


def get_points_covered_by_lidar_config(params, l_params, x, number_of_lidars):

    assert len(x) == 5*number_of_lidars

    linePoint = np.array([0, 0, 0])

    end_points = set()

    x_len = params["pom_x"]
    y_len = params["pom_y"]
    z_len = params["pom_z"]

    scale = params["scale"]

    num_x_cubes = x_len * scale
    num_y_cubes = y_len * scale
    num_z_cubes = z_len * scale

    resolution_x = x_len / scale
    resolution_y = y_len / scale
    resolution_z = z_len / scale

    yaws = range(0, 360)
    beam_angle = l_params["beam_angle"]

    planeNormals = get_normal_vectors()
    planePoints = get_points(num_x_cubes, num_y_cubes, num_z_cubes)
    for l in range(number_of_lidars):
        tx = x[l*5:(l+1)*5]

        tx = np.append(tx, 0)
        tx_no_rotate = np.zeros([6])
        for y in yaws:

            tx[5] = y

            assert len(tx) == 6

            for beam in beam_angle:
                tx[0] = (x[l * 5:(l + 1) * 5][0] - 30)
                tx[1] = (x[l * 5:(l + 1) * 5][1] - 20)
                tx[2] = (x[l * 5:(l + 1) * 5][2] - 2)
                directionVector = transform(get_direction_vector(y, beam), tx)[:3]
                tpn = transform(planeNormals, tx)[:3, :]
                tx[0] = (x[l*5:(l+1)*5][0] - 30) * scale
                tx[1] = (x[l*5:(l+1)*5][1] - 20) * scale
                tx[2] = (x[l*5:(l+1)*5][2] - 2) * scale
                tpp = transform(planePoints, tx)[:3, :]


                for i in range(6):
                    planeNormal = tpn[:, i]
                    planePoint = tpp[:, i]

                    intersectionPoint = get_intersection(
                        planePoint, planeNormal, linePoint, directionVector
                    )

                    if intersectionPoint is None:
                        continue
                    else:

                        intersectionPoint = intersectionPoint.astype(int)
                        if (
                            (-num_x_cubes / 2 <= intersectionPoint[0] <= num_x_cubes / 2)
                            and (
                                -num_y_cubes / 2 <= intersectionPoint[1] <= num_y_cubes / 2
                            )
                            and (
                                -num_z_cubes / 2 <= intersectionPoint[2] <= num_z_cubes / 2
                            )
                        ):
                            end_points.add(
                                f"{intersectionPoint[0]};{intersectionPoint[1]};{intersectionPoint[2]}"
                            )
            # Store all points in set that are less than distance beta

    points = list(end_points)
    points = [i.split(";") for i in points]
    points = np.array(points).astype(float).astype(int)

    reconstructed_points = []

    for point in points:
        reconstructed_points.extend(
            BresenhamInt3D(0, 0, 0, point[0], point[1], point[2])
        )

    a, b, c = [], [], []

    final_points = []
    for point in reconstructed_points:
        final_points.append((point[0] + 600, point[1] + 400, point[2] + 40))
    reconstructed_points = tuple(reconstructed_points)
    return reconstructed_points


if __name__ == "__main__":
    main()