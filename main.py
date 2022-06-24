from VD_Model import VDMCMA_Hyperparams, CentralisedAgents
from VD_Env import Switch, Universal_Env_Utility
from Training_Utility import Util


import random, numpy as np, torch
from pyvirtualdisplay import Display

#client file
def set_all_seeds(seed):
  random.seed(seed)
  # os.environ('PYTHONHASHSEED') = str(self.seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


#training 
SEED = 672
NUM_EPISODES = 400
set_all_seeds(SEED)
save_path = "C:/Users/Benjamin/Documents/Python/Trinity_Research/Reinforcement_Experimentation/Modular_VD/MVD_Models"
save_name = 'test_1'

'''I couldn't get this working on my local machine and the recording functionality doesn't work unless this does'''
'''display = Display(visible=0, size=(1400, 900))
display.start()'''
hyp = VDMCMA_Hyperparams()


#switch vid to true if Display import is working
env = Switch(num_agents=2,vid=False)
model = CentralisedAgents(hyp,state_dim=env.state_dim,action_dim=env.action_dim,n_agents=2)

'''
#simple training loop example
act = lambda o_s: model.act(o_s, True)
state1 = env.reset()
episode_count = 0

while episode_count < NUM_EPISODES:
  actions = act(state1)
  state2, reward, done, _ = env.step(actions)

  model.update(state1,actions,reward,state2,done)

  state1 = state2

  if np.all(done):
    state1 = env.reset()
    episode_count += 1
    if episode_count % 100 == 0:
      print(episode_count)

model.save(save_path,save_name)'''

RUNS = 3
env = Switch(num_agents=2)
state_dim = env.state_dim
action_dim = env.action_dim

train_log, train_reward_log, test_reward_log, agents, separated_agents = Util.run_experiments(Switch,
                                                               {'num_agents':2,'vid':False},
                                                               CentralisedAgents,
                                                               [hyp, state_dim, action_dim, 2],
                                                               n_episodes=NUM_EPISODES,
                                                               runs=RUNS,
                                                               verbose=True)

Util.plot_experiments(train_log, train_reward_log, test_reward_log, RUNS)

