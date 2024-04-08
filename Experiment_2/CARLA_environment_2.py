import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import cv2


class CarlaEnv(gym.Env):
    display_camera = True
    image_size = (240, 320, 3)
    camera_type = "semantic_segmentation"

    def __init__(self):
        # connect to carla server
        print("connecting to carla server")
        client = carla.Client("localhost", 2000)
        client.set_timeout(4.0)
        self.world = client.get_world()
        print("connected to carla server")

        # track the actors
        self.actor_list = []

        # setting the world settings
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = 0.2
        self.world.apply_settings(self.settings)

        # blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # ego vehicle  blueprint
        self.ego_vehicle_blueprint = blueprint_library.filter("model3")[0]

        # initialize the camera/display
        self.front_camera = None
        self.camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_blueprint = self.world.get_blueprint_library().find(
            "sensor.camera." + self.camera_type
        )
        self.camera_blueprint.set_attribute("image_size_x", f"{self.image_size[1]}")
        self.camera_blueprint.set_attribute("image_size_y", f"{self.image_size[0]}")
        self.camera_blueprint.set_attribute("fov", "110")
        # set the time in seconds between sensor captures
        self.camera_blueprint.set_attribute("sensor_tick", "0.1")

        # collision sensor blueprint
        self.collision_sensor_blueprint = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )

        # action space: throttle, steer, brake
        self.action_space = spaces.Discrete(36)  # 9 steering values and 4 throttle values

        # observation space: camera
        self.observation_space = spaces.Box(low=0, high=255, shape=self.image_size, dtype=np.uint8)

        # spawnpoints
        self.spawn_points = self.world.get_map().get_spawn_points()

    def reset(self, **kwargs):
        # cleanup
        for actor in self.actor_list:
            actor.destroy()
        cv2.destroyAllWindows()

        # reset the collision flag
        self.collision = False

        # track the actors
        self.actor_list = []

        # spawn the ego vehicle
        self._spawn_ego_vehicle()

        # attach and display camera
        self._attach_camera()
        while self.front_camera is None:
            time.sleep(0.01)
        if self.display_camera:
            cv2.namedWindow("Segmentation Camera", cv2.WINDOW_AUTOSIZE)
            self._display_camera()

        # attach collision sensor
        self._attatch_collision_sensor()

        self.time = time.time()
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        observation = self.front_camera

        return observation.astype(np.float32), {}

    def _spawn_ego_vehicle(self):
        # spawn the ego vehicle
        overlap = True
        while overlap:
            spawn_point = random.choice(self.spawn_points)
            overlap = False
            # check if the spawn point is too close to another actor
            for actor in self.actor_list:
                if spawn_point.location.distance(actor.get_location()) < 10:
                    overlap = True
                    break
        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_blueprint, spawn_point)
        self.actor_list.append(self.ego_vehicle)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

    def _attach_camera(self):
        self.camera = self.world.spawn_actor(
            self.camera_blueprint, self.camera_transform, attach_to=self.ego_vehicle
        )
        self.camera.listen(lambda image: self._process_image(image))
        self.actor_list.append(self.camera)

    def _display_camera(self):
        cv2.imshow("Segmentation Camera", self.front_camera)
        cv2.waitKey(1)

    def _process_image(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        im = np.array(image.raw_data)
        im = im.reshape((self.image_size[0], self.image_size[1], 4))[:, :, :3]  # ignore alpha channel
        self.front_camera = im

    def _attatch_collision_sensor(self):
        self.collision_sensor = self.world.spawn_actor(
            self.collision_sensor_blueprint, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_data(event))
        self.actor_list.append(self.collision_sensor)

    def _collision_data(self, event):
        self.collision = True

    def step(self, action):
        # map action to steer and throttle
        steer = action // 4
        throttle = action % 4

        # map steer to -1 to 1
        steer_values = [-0.9, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.9]
        steer = steer_values[steer]

        # map throttle and apply steer and throttle
        throttle_values = [0.0, 0.3, 0.7, 1.0]
        throttle = throttle_values[throttle]

        self.ego_vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle, steer=steer, brake=1.0 if throttle == 0 else 0.0
            )
        )

        # get the reward
        reward, done = self._get_reward()

        # get the next observation
        observation = self.front_camera

        # reshape the front_camera array
        observation = observation.reshape((240, 320, 3))

        # Display the camera
        if self.display_camera:
            self._display_camera()

        return observation, reward, done, False, {}

    def _get_reward(self):
        reward = 1
        done = False

        # Negative reward if collision
        if self.collision:
            reward = -1000
            done = True

        return reward, done
