from tqdm import tqdm
import numpy as np
from datetime import datetime

class Util:
    @staticmethod
    def run_experiments(Envir, EnvArgs, Agent, Hyp_,
                        n_episodes=1000, runs=10, log_interval=10, verbose=False, save_path = None):
        # store data for each run
        train_log = []
        train_reward_log = []
        test_reward_log = []


        if not save_path == None:
            save_path += "/{}".format(datetime.now())

        pbar = tqdm(range(runs))
        for count, i in enumerate(pbar):
            env = Envir(**EnvArgs)
            agents = Agent(*Hyp_)

            train_, train_reward = agents.run(env=env,
                                              train=True,
                                              exploration=True,
                                              max_episode=n_episodes,
                                              log_episode_interval=log_interval,
                                              verbose=verbose)
            print(train_reward[-1])

            # store result for every run
            train_log.append(train_)
            train_reward_log.append(train_reward)

            ####################################### TESTING ##############################

            test_, test_reward = agents.run(env=Envir(**EnvArgs),
                                            train=False,
                                            exploration=False,
                                            max_episode=10,
                                            log_episode_interval=1,
                                            verbose=verbose)
            # store result for every run
            test_reward_log.append(np.mean(test_reward))

            if not save_path == None:
                agents.save(save_path,"run_{}".format(count))

            pbar.set_description('r train: %.4f, r test: %.5f' % (train_reward[-1], test_reward_log[-1]))
        return train_log, train_reward_log, test_reward_log

    @staticmethod
    def plot_experiments(self, train_log, train_reward_log, test_reward_log, runs):
        ####################################### TRAINING #######################################
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111)

        for i in range(runs):
            ax.plot(train_log[i], train_reward_log[i], label=f'run {i + 1}')
        ax.set_title(f"Train learning Curve for {runs} runs")
        ax.set_ylabel("Episodic Reward")
        ax.set_xlabel("Iterations")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()