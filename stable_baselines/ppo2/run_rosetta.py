from stable_baselines.common.cmd_util import make_rosetta_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines import PPO2
from pyrosetta import init, pose_from_pdb, pose_from_sequence
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.ppo2.custom_policy import LstmCustomPolicy


def run():
    init()

    seed = 0
    # env = gym.make('gym_rosetta:protein-fold-v0')
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    env = make_rosetta_env("gym_rosetta:protein-fold-v0", 2, seed, use_subprocess=False)

    model = PPO2(LstmCustomPolicy, env, verbose=1, tensorboard_log='./log', n_steps=128, ent_coef=0.001, noptepochs=5,
                 nminibatches=16)
    model.learn(total_timesteps=900000)

    obs = env.reset()[0]
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    run()