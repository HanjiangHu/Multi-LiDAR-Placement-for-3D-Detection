import sys
import math

sys.path.append(
    "../../carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
)

import toml
import carla

from constants import CAMERA_IMAGE_X, CAMERA_IMAGE_Y, CAMERA_HEIGHT_POS

class LiDARSetup(object):
    def __init__(self, param_file, world, ego, callback_handler, setup=False):
        params = toml.load(f"../hyperparams/{param_file}")['lidar']
        self.world = world
        self.ego = ego
        self.callback_handler = callback_handler
        self.num = params["num"]
        self.channels = params["channels"]
        self.lower_fov = params["lower_fov"]
        self.upper_fov = params["upper_fov"]
        self.x = params["x"]
        self.y = params["y"]
        self.z = params["z"]
        self.roll = params["roll"]
        self.pitch = params["pitch"]
        self.stats = {}
        if setup:
            self._check_for_errors()
            self._setup()
        else:
            pass

    def _check_for_errors(self):
        assert self.num == len(self.x)
        assert self.num == len(self.y)
        assert self.num == len(self.z)
        assert self.num == len(self.roll)
        assert self.num == len(self.pitch)

    def _setup(self):
        self.stats["h"] = {}
        self.stats["t"] = {}
        self.stats["l"] = {}


        x, y, z, roll, pitch = (
            self.x[0],
            self.y[0],
            self.z[0],
            self.roll[0],
            self.pitch[0],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.stats['h'][0] = z

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.stats['t'][0] = lidar_transform

        lidar_bp = self.create_lidar(self.channels)

        self.stats['l'][0] = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego
        )


        self.stats['l'][0].listen(
            lambda point_cloud: self.callback_handler(point_cloud,
                                                        f"l_0",
                                                        self.callback_handler.queue)
        )

        print(
            f"Placed LiDAR 0 with channels {self.channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )

        x, y, z, roll, pitch = (
            self.x[1],
            self.y[1],
            self.z[1],
            self.roll[1],
            self.pitch[1],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.stats['h'][1] = z

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.stats['t'][1] = lidar_transform

        lidar_bp = self.create_lidar(self.channels)

        self.stats["l"][1] = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego
        )

        self.stats['l'][1].listen(
            lambda point_cloud: self.callback_handler(point_cloud,
                                                        f"l_1",
                                                        self.callback_handler.queue)
        )

        print(
            f"Placed LiDAR 1 with channels {self.channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )

        x, y, z, roll, pitch = (
            self.x[2],
            self.y[2],
            self.z[2],
            self.roll[2],
            self.pitch[2],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.stats["h"][2] = z

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.stats["t"][2] = lidar_transform

        lidar_bp = self.create_lidar()

        self.stats["l"][2] = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego
        )

        self.stats['l'][2].listen(
            lambda point_cloud: self.callback_handler(point_cloud,
                                                        f"l_2",
                                                        self.callback_handler.queue)
        )

        print(
            f"Placed LiDAR 2 with channels {self.channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )

        x, y, z, roll, pitch = (
            self.x[3],
            self.y[3],
            self.z[3],
            self.roll[3],
            self.pitch[3],
        )
        loc = carla.Location(x=x, y=y, z=z)
        self.stats["h"][3] = z

        rot = carla.Rotation(
            roll=int(roll * 180 / math.pi), pitch=int(pitch * 180 / math.pi)
        )
        lidar_transform = carla.Transform(loc, rot)
        self.stats["t"][3] = lidar_transform

        lidar_bp = self.create_lidar(self.channels)

        self.stats["l"][3] = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.ego
        )

        self.stats['l'][3].listen(
            lambda point_cloud: self.callback_handler(point_cloud,
                                                        f"l_3",
                                                        self.callback_handler.queue)
        )

        print(
            f"Placed LiDAR 3 with channels {self.channels} at x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}"
        )

    def create_lidar(self):
        print(f"Creating LIDAR with channels {self.channels}")
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("noise_stddev", "0.2")

        lidar_bp.set_attribute("upper_fov", str(self.upper_fov))
        lidar_bp.set_attribute("lower_fov", str(self.lower_fov))
        lidar_bp.set_attribute("channels", str(self.channels))
        lidar_bp.set_attribute("range", str(100.0))
        lidar_bp.set_attribute("rotation_frequency", str(2.0 / 0.1))
        points = 20_000 * self.channels
        lidar_bp.set_attribute("points_per_second", str(points))
        return lidar_bp

    def process_render_object(self, instance, data):
        data["points"][instance[1]] = instance[0]
        return data

    def destroy(self):
        for i in range(self.num):
            self.stats['l'][i].destroy()

    def create_lidar_spec(self):
        """
        Define the sensors spec as required by the Scenario Runner

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        specs = []
        for i in range(self.num):
            spec = {}
            spec['x'], spec['y'], spec['z'] = self.x[i], self.y[i], self.z[i]
            spec['roll'], spec['pitch'], spec['yaw'] = self.roll[i], self.pitch[i], 0.0

            spec["noise_stddev"] = "0.2"

            spec["upper_fov"] = str(self.upper_fov)
            spec["lower_fov"] = str(self.lower_fov)
            spec["channels"] = str(self.channels)
            spec["range"] = str(100.0)
            spec["rotation_frequency"] = str(2.0 / 0.1) ###!!!!10Hz
            points = 5_000 * self.channels
            spec["points_per_second"] =  str(points)
            spec["id"] = f"l_{i}"
            spec["type"] = "sensor.lidar.ray_cast"

            specs.append(spec)

        return specs


class Sensor(object):
    def __init__(self, lidars):
        self.lidars = lidars

    def create_sensor_spec(self):
        """
        Define the sensors spec as required by the Scenario Runner

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """

        specs = self.lidars.create_lidar_spec()
        camera_spec = {}
        camera_spec['width'] = str(CAMERA_IMAGE_X) 
        camera_spec['height'] = str(CAMERA_IMAGE_Y) 
        camera_spec["fov"] = str(90.0)
        camera_spec['id'] = "camera01"
        camera_spec['x'], camera_spec['y'], camera_spec['z'] = 0.0, 0.0, CAMERA_HEIGHT_POS
        camera_spec['roll'], camera_spec['pitch'], camera_spec['yaw'] = 0.0, 0.0, 0.0
        camera_spec['type'] = 'sensor.camera.rgb'

        depth_camera_spec = {}
        depth_camera_spec["width"] = str(CAMERA_IMAGE_X)
        depth_camera_spec["height"] = str(CAMERA_IMAGE_Y)
        depth_camera_spec["fov"] = str(90.0)
        depth_camera_spec['id'] = "depth01"
        depth_camera_spec['x'], depth_camera_spec['y'], depth_camera_spec['z'] = 0.0, 0.0, CAMERA_HEIGHT_POS
        depth_camera_spec['roll'], depth_camera_spec['pitch'], depth_camera_spec['yaw'] = 0.0, 0.0, 0.0
        depth_camera_spec['type'] = "sensor.camera.depth"

        specs.extend([camera_spec, depth_camera_spec])

        return specs


