import gym
from VD_Model import VDMCMA_Hyperparams, CentralisedAgents
from VD_Env import Switch, Universal_Env_Utility, Env_wrapper_VD
from Training_Utility import Util

#class mountain_car_env_wrapper


env_name = 'MountainCar-v0'
temp_env = Env_wrapper_VD(env_name)
state_dim =  temp_env.state_dim
action_dim = temp_env.action_dim

RUNS = 1
NUM_EPISODES = 1000
save_path = "C:/Users/Benjamin/Documents/Python/Trinity_Research/Reinforcement_Experimentation/Github Repositories/Modular_VD/Saved Models"
hyp = VDMCMA_Hyperparams()
train_log, train_reward_log, test_reward_log= Util.run_experiments(Env_wrapper_VD,
                                                               {'env_name':env_name,'vid':False},
                                                               CentralisedAgents,
                                                               [hyp, state_dim, action_dim, 1],
                                                               n_episodes=NUM_EPISODES,
                                                               runs=RUNS,
                                                               verbose=True,
                                                               save_path=save_path)