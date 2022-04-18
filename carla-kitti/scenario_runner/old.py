def process_multilidar_data(self, data, datapoints):
    points = []
    for i in range(6):
        point_cloud = np.copy(
            np.frombuffer(data["points"][i].raw_data, dtype=np.dtype("f4"))
        )
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))[
            :, :-1
        ]

        point = np.append(point_cloud, np.ones((point_cloud.shape[0], 1)), axis=1)
        # Point transformation

        transform_matrix = self.lidar_transforms[i].get_matrix()  # LiDAR to Car

        point_wrt_car = np.dot(transform_matrix, point.T).T

        point = point_wrt_car[:, :-1]

        point[:, 2] -= self.heights[i]
        points.append(point)

    self._save_training_files(data["image"], datapoints, np.vstack(points))
    self.save_no += 1


def process_single_lidar_data(self, data, datapoints):
    point_cloud = np.copy(np.frombuffer(data["points"].raw_data, dtype=np.dtype("f4")))
    point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))[:, :-1]


    points = np.append(point_cloud, np.ones((point_cloud.shape[0], 1)), axis=1)
    # Point transformation

    transform_matrix = self.lidar_to_car_transform.get_matrix()  # LiDAR to Car

    point_wrt_car = np.dot(transform_matrix, points.T).T

    points = point_wrt_car[:, :-1]

    points[:, 2] -= LIDAR_HEIGHT_POS

    self._save_training_files(data["image"], datapoints, points)
    self.save_no += 1


def _attach_lidar_with_location_to_ego_vehicle(self, location):
    lidar_bp = self.create_lidar()

    lidar_transform = carla.Transform(location)



    lidar = self.world.spawn_actor(
        lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
    )
    self.lidars.append(lidar)

    lidar.listen(lambda point_cloud: self.lidar_callback(point_cloud))


def _attach_multiple_lidars_to_ego_vehicle(self):
    self.lidars = []
    loc1 = carla.Location(x=-0.5, y=-0.5, z=LIDAR_HEIGHT_POS)
    loc2 = carla.Location(x=0.5, y=-0.5, z=LIDAR_HEIGHT_POS)
    loc3 = carla.Location(x=-0.5, y=0.5, z=LIDAR_HEIGHT_POS)
    loc4 = carla.Location(x=0.5, y=0.5, z=LIDAR_HEIGHT_POS)

    self.lidar_transforms = []
    lidar_transform1 = carla.Transform(loc1)
    lidar_transform2 = carla.Transform(loc2)
    lidar_transform3 = carla.Transform(loc3)
    lidar_transform4 = carla.Transform(loc4)
    self.lidar_transforms.extend(
        [lidar_transform1, lidar_transform2, lidar_transform3, lidar_transform4]
    )

    self.heights = [
        LIDAR_HEIGHT_POS,
        LIDAR_HEIGHT_POS,
        LIDAR_HEIGHT_POS,
        LIDAR_HEIGHT_POS,
    ]
    channels = 4.0

    lidar_bp = self.create_lidar(channels, 4)

    lidar1 = self.world.spawn_actor(
        lidar_bp, lidar_transform1, attach_to=self.ego_vehicles[0]
    )
    self.lidars.append(lidar1)
    lidar1.listen(
        lambda point_cloud: self.multiple_lidar_callback(point_cloud, "lidar01")
    )

    lidar2 = self.world.spawn_actor(
        lidar_bp, lidar_transform2, attach_to=self.ego_vehicles[0]
    )
    self.lidars.append(lidar2)
    lidar2.listen(
        lambda point_cloud: self.multiple_lidar_callback(point_cloud, "lidar02")
    )

    lidar3 = self.world.spawn_actor(
        lidar_bp, lidar_transform3, attach_to=self.ego_vehicles[0]
    )
    self.lidars.append(lidar3)
    lidar3.listen(
        lambda point_cloud: self.multiple_lidar_callback(point_cloud, "lidar03")
    )

    lidar4 = self.world.spawn_actor(
        lidar_bp, lidar_transform4, attach_to=self.ego_vehicles[0]
    )
    self.lidars.append(lidar4)

    lidar4.listen(
        lambda point_cloud: self.multiple_lidar_callback(point_cloud, "lidar04")
    )


def _attach_multiple_lidars_with_multiple_beams_to_ego_vehicle_custom_orientation(self):
    self._attach_4_16_lidars()


def _attach_4_16_lidars(self):

    self._attach_single_lidar()


def _attach_single_lidar(self):
    callback = {
        0: self.callback1,
        1: self.callback2,
        2: self.callback3,
        3: self.callback4,
    }
    self.lidars, self.positions, self.lidar_transforms = [], {}, {}
    channels = 16
    i = 0
    x, y, z, roll, pitch = (
        self.params["1"]["x"][i],
        self.params["1"]["y"][i],
        self.params["1"]["z"][i],
        self.params["1"]["roll"][i],
        self.params["1"]["pitch"][i],
    )
    loc = carla.Location(x=x, y=y, z=z)
    self.positions[i] = z

    rot = carla.Rotation(
        roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
    )
    lidar_transform = carla.Transform(loc, rot)
    self.lidar_transforms[i] = lidar_transform

    lidar_bp = self.create_lidar(channels, i)

    lidar = self.world.spawn_actor(
        lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
    )
    lidar.listen(lambda point_cloud: self.callback1(point_cloud, "lidar_0"))
    self.lidars.append(lidar)
    print(
        f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
    )

    i = 1
    x, y, z, roll, pitch = (
        self.params["1"]["x"][i],
        self.params["1"]["y"][i],
        self.params["1"]["z"][i],
        self.params["1"]["roll"][i],
        self.params["1"]["pitch"][i],
    )
    loc2 = carla.Location(x=x, y=y, z=z)
    self.positions[i] = z

    rot2 = carla.Rotation(
        roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
    )
    lidar_transform2 = carla.Transform(loc2, rot2)
    self.lidar_transforms[i] = lidar_transform2

    lidar_bp2 = self.create_lidar(channels, i)

    lidar2 = self.world.spawn_actor(
        lidar_bp2, lidar_transform2, attach_to=self.ego_vehicles[0]
    )

    lidar2.listen(lambda point_cloud: self.callback2(point_cloud, "lidar_1"))
    self.lidars.append(lidar2)
    print(lidar, lidar2)
    print(
        f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
    )


def _attach_6_10_lidar(self):
    self.lidars_4_10, self.heights_4_10, self.lidar_transforms_4_10 = [], [], []

    channels = 10

    for i in range(12):
        x, y, z, roll, pitch = (
            self.params["12_10"]["x"][i],
            self.params["12_10"]["y"][i],
            self.params["12_10"]["z"][i],
            self.params["12_10"]["roll"][i],
            self.params["12_10"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4_10.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4_10.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4_10.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_120_{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )


def _attach_40_lidar(self):
    self.lidars_4_10, self.heights_4_10, self.lidar_transforms_4_10 = [], [], []

    channels = 40

    for i in range(1):
        x, y, z, roll, pitch = (
            self.params["40"]["x"][i],
            self.params["40"]["y"][i],
            self.params["40"]["z"][i],
            self.params["40"]["roll"][i],
            self.params["40"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4_10.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4_10.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4_10.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_40_{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )


def _attach_4_10_lidars(self):
    self.lidars_4_10, self.heights_4_10, self.lidar_transforms_4_10 = [], [], []

    channels = 10

    for i in range(4):
        x, y, z, roll, pitch = (
            self.params["4_10"]["x"][i],
            self.params["4_10"]["y"][i],
            self.params["4_10"]["z"][i],
            self.params["4_10"]["roll"][i],
            self.params["4_10"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4_10.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4_10.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4_10.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_4_10_{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )


def _attach_argo_lidars(self):
    self.lidars_4_argo, self.heights_4_argo, self.lidar_transforms_4_argo = [], [], []

    channels = 16

    for i in range(4):
        x, y, z, roll, pitch = (
            self.params["lidar_argo"]["x"][i],
            self.params["lidar_argo"]["y"][i],
            self.params["lidar_argo"]["z"][i],
            self.params["lidar_argo"]["roll"][i],
            self.params["lidar_argo"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4_argo.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4_argo.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4_argo.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_argo_{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_cruise_lidars(self):
    self.lidars_4_cruise, self.heights_4_cruise, self.lidar_transforms_4_cruise = (
        [],
        [],
        [],
    )

    channels = 16

    for i in range(4):
        x, y, z, roll, pitch = (
            self.params["lidar_cruise"]["x"][i],
            self.params["lidar_cruise"]["y"][i],
            self.params["lidar_cruise"]["z"][i],
            self.params["lidar_cruise"]["roll"][i],
            self.params["lidar_cruise"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4_cruise.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4_cruise.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4_cruise.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_cruise_{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_random2_lidars(self):
    self.lidars_4_random2, self.heights_4_random2, self.lidar_transforms_4_random2 = (
        [],
        [],
        [],
    )
    channels = 16

    for i in range(4):
        x, y, z, roll, pitch = (
            self.params["lidar_random2"]["x"][i],
            self.params["lidar_random2"]["y"][i],
            self.params["lidar_random2"]["z"][i],
            self.params["lidar_random2"]["roll"][i],
            self.params["lidar_random2"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4_random2.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4_random2.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4_random2.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_random2_{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_multiple_lidars_to_ego_vehicle_custom_orientation(self):
    self.lidars, self.heights, self.lidar_transforms = [], [], []

    channels = self.params["lidar"]["channels"]

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["positions"]["x"][i],
            self.params["positions"]["y"][i],
            self.params["positions"]["z"][i],
            self.params["positions"]["roll"][i],
            self.params["positions"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(point_cloud, f"lidar0{i}")
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_2_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_2, self.heights_2, self.lidar_transforms_2 = [], [], []

    channels = 2

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_2"]["x"][i],
            self.params["lidar_2"]["y"][i],
            self.params["lidar_2"]["z"][i],
            self.params["lidar_2"]["roll"][i],
            self.params["lidar_2"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_2.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_2.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_2.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_2_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_3_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_3, self.heights_3, self.lidar_transforms_3 = [], [], []

    channels = 3

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_3"]["x"][i],
            self.params["lidar_3"]["y"][i],
            self.params["lidar_3"]["z"][i],
            self.params["lidar_3"]["roll"][i],
            self.params["lidar_3"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_3.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_3.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_3.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_3_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_4_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_4, self.heights_4, self.lidar_transforms_4 = [], [], []

    channels = 4

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_4"]["x"][i],
            self.params["lidar_4"]["y"][i],
            self.params["lidar_4"]["z"][i],
            self.params["lidar_4"]["roll"][i],
            self.params["lidar_4"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_4.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_4.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_4.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_4_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_5_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_5, self.heights_5, self.lidar_transforms_5 = [], [], []

    channels = 5

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_5"]["x"][i],
            self.params["lidar_5"]["y"][i],
            self.params["lidar_5"]["z"][i],
            self.params["lidar_5"]["roll"][i],
            self.params["lidar_5"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_5.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_5.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_5.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_5_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_6_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_6, self.heights_6, self.lidar_transforms_6 = [], [], []

    channels = 6

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_6"]["x"][i],
            self.params["lidar_6"]["y"][i],
            self.params["lidar_6"]["z"][i],
            self.params["lidar_6"]["roll"][i],
            self.params["lidar_6"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_6.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_6.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_6.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_6_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_7_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_7, self.heights_7, self.lidar_transforms_7 = [], [], []

    channels = 7

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_7"]["x"][i],
            self.params["lidar_7"]["y"][i],
            self.params["lidar_7"]["z"][i],
            self.params["lidar_7"]["roll"][i],
            self.params["lidar_7"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_7.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_7.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_7.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_7_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_12_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_12, self.heights_12, self.lidar_transforms_12 = [], [], []

    channels = 12

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_12"]["x"][i],
            self.params["lidar_12"]["y"][i],
            self.params["lidar_12"]["z"][i],
            self.params["lidar_12"]["roll"][i],
            self.params["lidar_12"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_12.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_12.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_12.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_12_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )


def _attach_4_8_lidars_to_ego_vehicle_custom_orientation(self):

    self.lidars_8, self.heights_8, self.lidar_transforms_8 = [], [], []

    channels = 8

    for i in range(self.number_of_lidars):
        x, y, z, roll, pitch = (
            self.params["lidar_8"]["x"][i],
            self.params["lidar_8"]["y"][i],
            self.params["lidar_8"]["z"][i],
            self.params["lidar_8"]["roll"][i],
            self.params["lidar_8"]["pitch"][i],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.heights_8.append(z)

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.lidar_transforms_8.append(lidar_transform)

        lidar_bp = self.create_lidar(channels, i)

        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego_vehicles[0]
        )
        self.lidars_8.append(lidar)
        lidar.listen(
            lambda point_cloud: self.multiple_lidar_callback(
                point_cloud, f"lidar_8_0{i}"
            )
        )

        print(
            f"Placed LiDAR {i} with channels {channels} at x={x}, y={y}, z={z}, roll={int(roll*180/math.pi)}, pitch={int(pitch*180/math.pi)}"
        )
