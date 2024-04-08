from stable_baselines3 import DQN, PPO
import os
from CARLA_environment_5 import CarlaEnv
import time

print("setting folders for logs and models")
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

print("connecting to env..")
env = CarlaEnv()
env.reset()

print("creating model..")
# model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.001, tensorboard_log=logdir, device="auto")
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    tensorboard_log=logdir,
    device="auto",
    buffer_size=10_000,
)

TIMESTEPS = 100_000
iters = 0
while iters < 5:
    iters += 1
    print("Iteration ", iters, " is to commence...")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
    print("Iteration ", iters, " has been trained")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
