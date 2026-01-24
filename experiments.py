import numpy as np
import matplotlib.pyplot as plt
import time
from linnash import simulate_linnash
from fairLinPE import simulate_fairLinPE
from fairLinUCB import simulate_fairLinUCB



import os

cached_dir = "caches"
if not os.path.exists(cached_dir):
    os.makedirs(cached_dir)

plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)



class LinearBanditEnvironment:


    def __init__(self, X, theta_star, noise_std=0.1, seed=42):
        self.rng = np.random.default_rng(seed)
        self.arms = X
        self.theta_star = theta_star
        self.noise_std = noise_std
        self.num_arms, self.d = X.shape
        self.mean_rewards = X @ theta_star
        
    def get_reward(self, arm_idx):
        expected_reward = self.mean_rewards[arm_idx]
        noise = self.rng.normal(0, self.noise_std)
        return expected_reward + noise






def run_all_algorithms(env, X, T, num_trials, sigma2, d, dataset):

    # USE np.load() to load cached values instead of running again

    regret_linnash = simulate_linnash(env, X, T, num_trials=num_trials, sigma2=sigma2)
    np.save(f"{cached_dir}/regret_linnash_{dataset}_d_{d}_.npy", regret_linnash)

    regret_fairLinPE = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=0)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_d_{d}.npy", regret_fairLinPE)

    regret_fairLinUCB = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=0)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_d_{d}.npy", regret_fairLinUCB)

    title = f"MSLR-WEB10K, d={d}" if dataset == "mslr" else (f"Yahoo! LTRC, d={d}" if dataset == "yahoo" else "Synthetic Dataset")

    plt.figure()
    plt.plot(regret_fairLinPE, label=r"FairLinPE", marker='o', markevery = T/10)
    plt.plot(regret_fairLinUCB, label=r"FairLinUCB", marker='*', markevery = T/10)
    plt.plot(regret_linnash, label=r"LinNash", marker='P', markevery = T/10)
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(r"Nash Regret", fontsize=20)
    plt.title(title, fontsize=20)        
    plt.legend(fontsize=20, columnspacing=0.1, handletextpad=0.1, labelspacing=0.1, borderpad=0.1 , framealpha=1, ncols = 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/all_algorithms_d_{d}_T_{T}_{dataset}.png")
    plt.close()


def run_compare_PE_UCB_vary_p(env, X, T, num_trials, sigma2, d, dataset, p=0):

    # USE np.load() to load cached values instead of running again


    regret_fairLinPE = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=p)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_p_{p_}.npy", regret_fairLinPE)
    
    regret_fairLinUCB = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=p)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_p_{p_}.npy", regret_fairLinUCB)

    title = f"MSLR-WEB10K, d={d}" if dataset == "mslr" else (f"Yahoo! LTRC, d={d}" if dataset == "yahoo" else "Synthetic Dataset")


    plt.figure()
    plt.plot(regret_fairLinPE, label=r"FairLinPE", marker='o', markevery = T/10)
    plt.plot(regret_fairLinUCB, label=r"FairLinUCB", marker='P', markevery = T/10)
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(r"$p$-mean Regret", fontsize=20)
    plt.title(title, fontsize=20)         
    plt.legend(fontsize=20, columnspacing=0.1, handletextpad=0.1, labelspacing=0.1, borderpad=0.1 , framealpha=1, ncols = 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/p_comparison_PE_UCB_p_{p}_{dataset}.png")
    plt.close()


def run_PE_vary_p(env, X, T, num_trials, sigma2, d, dataset):


    # USE np.load() to load cached values instead of running again

    regret_fairLinPE_p1 = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=1)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_p_1.npy", regret_fairLinPE_p1)

    regret_fairLinPE_p0 = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=0)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_p_0.npy", regret_fairLinPE_p0)

    regret_fairLinPE_pm1 = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=-1)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_p_m1.npy", regret_fairLinPE_pm1)

    regret_fairLinPE_pm2 = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=-2)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_p_m2.npy", regret_fairLinPE_pm2)

    regret_fairLinPE_pm5 = simulate_fairLinPE(env, X, T, num_trials=num_trials, sigma2=sigma2, p=-5)
    np.save(f"{cached_dir}/regret_fairLinPE_{dataset}_p_m5.npy", regret_fairLinPE_pm5)

    title = f"MSLR-WEB10K, d={d}" if dataset == "mslr" else (f"Yahoo! LTRC, d={d}" if dataset == "yahoo" else "Synthetic Dataset")

    plt.figure()
    plt.plot(regret_fairLinPE_p1, label=r"$p=1$", marker='o', markevery = T/10)
    plt.plot(regret_fairLinPE_p0, label=r"$p=0$", marker='P', markevery = T/10)
    plt.plot(regret_fairLinPE_pm1, label=r"$p=-1$", marker='*', markevery = T/10)
    plt.plot(regret_fairLinPE_pm2, label=r" $p=-2$", marker='|', markevery = T/10)
    plt.plot(regret_fairLinPE_pm5, label=r"$p=-5$", marker='^', markevery = T/10)


    
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(r"$p$-mean Regret", fontsize=20)
    plt.title(title, fontsize=20)           # change plot title based on dataset choosen 
    plt.legend(fontsize=20, columnspacing=0.1, handletextpad=0.1, labelspacing=0.1, borderpad=0.1 , framealpha=1, ncols = 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/FairLinPE_vary_p_d_{d}.png")
    plt.close()


def run_UCB_vary_p(env, X, T, num_trials, sigma2, d, dataset):

    # USE np.load() to load cached values instead of running again

    regret_fairLinUCB_p1 = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=1)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_p_1.npy", regret_fairLinUCB_p1)

    regret_fairLinUCB_p0 = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=0)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_p_1.npy", regret_fairLinUCB_p0)

    regret_fairLinUCB_pm1 = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=-1)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_p_1.npy", regret_fairLinUCB_pm1)

    regret_fairLinUCB_pm2 = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=-2)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_p_1.npy", regret_fairLinUCB_pm2)

    regret_fairLinUCB_pm5 = simulate_fairLinUCB(env, X, T, num_trials=num_trials, sigma2=sigma2, p=-5)
    np.save(f"{cached_dir}/regret_fairLinUCB_{dataset}_p_1.npy", regret_fairLinUCB_pm5)

    title = f"MSLR-WEB10K, d={d}" if dataset == "mslr" else (f"Yahoo! LTRC, d={d}" if dataset == "yahoo" else "Synthetic Dataset")

    plt.figure()
    plt.plot(regret_fairLinUCB_p1, label=r"$p=1$", marker='o', markevery = T/10)
    plt.plot(regret_fairLinUCB_p0, label=r"$p=0$", marker='P', markevery = T/10)
    plt.plot(regret_fairLinUCB_pm1, label=r"$p=-1$", marker='|', markevery = T/10)
    plt.plot(regret_fairLinUCB_pm2, label=r"$p=-2$", marker='', markevery = T/10)
    plt.plot(regret_fairLinUCB_pm5, label=r"$p=-5$", marker='^', markevery = T/10)


    
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(r"$p$-mean Regret", fontsize=20)
    plt.title(title, fontsize=20)           # change plot title based on dataset choosen 
    plt.legend(fontsize=20, columnspacing=0.1, handletextpad=0.1, labelspacing=0.1, borderpad=0.1 , framealpha=1, ncols = 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/FairLinUCB_vary_p_d_{d}_{dataset}.png")
    plt.close()




