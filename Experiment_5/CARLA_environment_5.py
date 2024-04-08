import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import cv2
import math

class CarlaEnv(gym.Env):
    display_camera = True
    image_size = (240, 320, 3)
    camera_type = "semantic_segmentation"  # rgb

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
        self.camera_blueprint.set_attribute("sensor_tick", "0.05")

        # collision sensor blueprint
        self.collision_sensor_blueprint = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )

        # lane invasion sensor blueprint
        self.lane_invasion_sensor_blueprint = self.world.get_blueprint_library().find(
            "sensor.other.lane_invasion"
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

        # reset the lane invasion flag
        self.lane_invasion = False

        # track the actors
        self.actor_list = []

        # spawn the ego vehicle
        self._spawn_ego_vehicle()

        # set and get the destination
        self._set_destination()
        self.distance = self._get_distance_to_destination()
        self.improvement = 0

        # set the speed
        self.speed = 0

        # attach and display camera
        self._attach_camera()
        while self.front_camera is None:
            time.sleep(0.01)
        if self.display_camera:
            cv2.namedWindow("Segmentation Camera", cv2.WINDOW_AUTOSIZE)
            self._display_camera()

        # attach collision sensor
        self._attatch_collision_sensor()

        # attach the lane invasion sensor
        self._attach_lane_invasion_sensor()

        self.time = time.time()
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        observation = self.front_camera

        return observation, {}

    def _spawn_ego_vehicle(self):
        # spawn the ego vehicle
        spawn_point = random.choice(self.spawn_points)
        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_blueprint, spawn_point)
        self.actor_list.append(self.ego_vehicle)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # print(f"spawned ego vehicle at {spawn_point.location}")

    def _set_destination(self):
        self.destination = random.choice(self.spawn_points)
        # print(f"destination set to {self.destination.location}")

    def _get_distance_to_destination(self):
        return self.ego_vehicle.get_location().distance(self.destination.location)
        # print(f"distance to destination: {self.distance}")

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
        im = im.reshape((self.image_size[0], self.image_size[1], 4))[
            :, :, :3
        ]  # ignore alpha channel

        improvement_color = [
            [0, 0, 255],  # red
            [255, 0, 0],  # yellow
            [0, 255, 0],  # green
        ]
        im[-1, :, :] = improvement_color[self.improvement]

        speed_color = [
            [0, 0, 255],  # red
            [255, 0, 0],  # yellow
            [0, 255, 0],  # green
        ]
        speed_index = lambda x: 0 if x < 5 else 1 if x < 10 else 2
        im[0, :, :] = speed_color[speed_index(self.speed)]

        # add a row to represent the direction
        self.front_camera = im

    def _attatch_collision_sensor(self):
        self.collision_sensor = self.world.spawn_actor(
            self.collision_sensor_blueprint, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self._collision_data(event))
        self.actor_list.append(self.collision_sensor)

    def _collision_data(self, event):
        self.collision = True

    def _attach_lane_invasion_sensor(self):
        self.lane_invasion_sensor = self.world.spawn_actor(
            self.lane_invasion_sensor_blueprint, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self._lane_invasion_data(event))
        self.actor_list.append(self.lane_invasion_sensor)

    def _lane_invasion_data(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        for lane_type in lane_types:
            if (lane_type == carla.LaneMarkingType.Solid): #or lane_type == carla.LaneMarkingType.SolidSolid):
                self.lane_invasion = True
                # print("Lane Invasion!")
                break

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

        # get the improvement
        distance = self._get_distance_to_destination()
        if distance < self.distance - 0.8:
            self.improvement = 2
        elif distance < self.distance - 0.3:
            self.improvement = 1
        else:
            self.improvement = 0
        self.distance = distance
        # print(f"Distance to destination: {self.distance}")

        # vehicle speed
        v = self.ego_vehicle.get_velocity()
        self.speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

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
        done = False
        reward = 0

        # Negative reward if collision
        if self.collision:
            reward -= 1000
            done = True
            return reward, done

        # Negative reward if lane invasion
        if self.lane_invasion:
            reward -= 500
        else:
            # Positive reward for speed in lane
            v = self.ego_vehicle.get_velocity()
            kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            if kmh > 10:
                reward += 10
            elif kmh > 5:
                reward += 5
            else:
                reward += 0

        # Positive reward if destination reached
        if self.distance < 5:
            reward += 500
            done = True
            print("Destination Reached!")
        elif self.improvement == 2:
            reward += 10
        elif self.improvement == 1:
            reward += 5
        elif self.improvement == 0:
            reward += 0

        return reward, done
