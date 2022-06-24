import gym, glob, io, base64
from ma_gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay
import numpy as np



class Switch:
    ''' wrapper to encode the time in the state '''

    def __init__(self, num_agents=2, vid=False):
        nm = "Switch%d-v0" % num_agents
        self.n_agents = num_agents
        self.env = Universal_Env_Utility.wrap_env(gym.make(nm)) if vid else gym.make(nm)
        self.t, self.dt = 0, 1  # 0.01
        self.render, self.close = self.env.render, self.env.close
        # self.conv_state = lambda obs: [o+[self.t] for o in obs]

        self.ts = np.linspace(0, np.pi, 30)
        self.conv_state = lambda obs: [o + np.cos(self.t * self.ts).tolist() for o in obs]
        self.state_dim = len(self.ts) + 2
        self.action_dim = 5

    def reset(self):
        return self.conv_state(self.env.reset())

    def step(self, a):
        obs_n, reward_n, done_n, info = self.env.step(a)
        self.t += self.dt
        if np.all(done_n): self.t = 0  #  switch has a max of 100 steps
        return self.conv_state(obs_n), reward_n, done_n, info


class Universal_Env_Utility:
    @staticmethod
    def record_ep(agents, env, maxT=100):

        def show_video():
            mp4list = glob.glob('video/*.mp4')
            if len(mp4list) > 0:
                mp4 = mp4list[0]
                video = io.open(mp4, 'r+b').read()
                encoded = base64.b64encode(video)
                ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                </video>'''.format(encoded.decode('ascii'))))
            else:
                print("Could not find video")

        obs_n = env.reset()
        env.render()
        for _ in range(maxT):
            if type(agents) == list:
                a = np.array([agent.act(state) for state, agent in zip(obs_n, agents)])
            else:
                a = agents.act(obs_n)
            obs_n, reward_n, done_n, info = env.step(a)
            env.render()
            if np.all(done_n): break
        env.close()
        show_video()
    @staticmethod
    def wrap_env(env):
        env = Monitor(env, './video', force=True)
        return env


