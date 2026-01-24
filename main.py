import numpy as np
from data_loader import generate_synthetic_data_with_lasso, load_and_process_mslr, load_and_process_yahoo_tar
from experiments import LinearBanditEnvironment, run_all_algorithms, run_compare_PE_UCB_vary_p, run_PE_vary_p, run_UCB_vary_p
import matplotlib.pyplot as plt


if __name__ == "__main__":

    d = 10        
    T = 10000 
    sigma2 = 0.5    
    num_trials = 1
    num_arms = 1000
    dataset = "mslr"         # Change to "mslr" or "yahoo" or "synthetic"


    np.random.seed(42)



    if (dataset == "mslr"):
        file_path = "/home/nishant/linwelfare/LinNash/MSLR-WEB10K/Fold1/train.txt"              # change to your path to mslr -> train.txt
        X, theta_star = load_and_process_mslr(file_path, d_dim=d)
    elif (dataset == "yahoo"):
        file_path = "/home/nishant/linwelfare/LinNash/YahooLTRC/ltrc_yahoo.tar.bz2"            # change to your path to ltrc_yahoo.tar.bz2 (ensure that you are using this version of the dataset)
        X, theta_star = load_and_process_yahoo_tar(file_path, d_dim=d)
    elif dataset == "synthetic":
        X, theta_star = generate_synthetic_data_with_lasso(n_arms=num_arms, d_dim=d)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")



    env = LinearBanditEnvironment(X, theta_star, noise_std=np.sqrt(sigma2))             # environment


# ***************** Experiment (a) : Comparing all algorithms. Vary T and dataset for different plots ******************************
    
    run_all_algorithms(env, X, T, num_trials, sigma2, d, dataset)      

# ********************************************************************************************************************************** 




# ***************** Experiment (b) : Comparing FairLinPE and FairLinUCB for different p ******************************

    p_values = [0.5, -0.5, -1.5]

    for p_ in p_values:
        run_compare_PE_UCB_vary_p(env, X, T, num_trials, sigma2, d, dataset, p=p_)


# **********************************************************************************************************************************





# ***************** Experiment (c) : Plotting FairLinPE and FairLinUCB individually for different p ******************************

    run_PE_vary_p(env, X, T, num_trials, sigma2, d, dataset)
    run_UCB_vary_p(env, X, T, num_trials, sigma2, d, dataset)


# **********************************************************************************************************************************




# ***************** Experiment (d) : Plotting FairLinPE and FairLinUCB individually for different d ******************************



    T = 10000
    dataset = "synthetic"
    d_values = [6, 9, 12, 15, 18]

    for d_ in d_values:
        if (dataset == "mslr"):
            file_path = "/home/nishant/linwelfare/LinNash/MSLR-WEB10K/Fold1/train.txt"              # add your path to mslr -> train.txt
            X, theta_star = load_and_process_mslr(file_path, d_dim=d)
        elif (dataset == "yahoo"):
            file_path = "/home/nishant/linwelfare/LinNash/YahooLTRC/ltrc_yahoo.tar.bz2"            # add your path to ltrc_yahoo.tar.bz2 (ensure that you are using this version of the dataset)
            X, theta_star = load_and_process_yahoo_tar(file_path, d_dim=d)
        elif dataset == "synthetic":
            X, theta_star = generate_synthetic_data_with_lasso(n_arms=num_arms, d_dim=d)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        env = LinearBanditEnvironment(X, theta_star, noise_std=np.sqrt(sigma2))             # environment

        run_all_algorithms(env, X, T, num_trials, sigma2, d_, dataset)



    plt.figure()
    for d_ in d_values:
        regret = np.load(f"cached/regret_fairLinPE_{dataset}_d_{d_}.npy")

        plt.plot(regret, label=rf"$d={d_}$", marker='o', markevery = T/10)


    title = f"MSLR-WEB10K, d={d}" if dataset == "mslr" else (f"Yahoo! LTRC, d={d}" if dataset == "yahoo" else "Synthetic Dataset")

    
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(r"Nash Regret", fontsize=20)
    plt.title(title, fontsize=20)           # change plot title based on dataset choosen 
    plt.legend(fontsize=20, columnspacing=0.1, handletextpad=0.1, labelspacing=0.1, borderpad=0.1 , framealpha=1, ncols = 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/FairLinPE_vary_d_{dataset}.png")
    plt.close()



    plt.figure()
    for d_ in d_values:
        regret = np.load(f"cached/regret_fairLinUCB_{dataset}_d_{d_}.npy")

        plt.plot(regret, label=rf"$d={d_}$", marker='o', markevery = T/10)

    
    plt.xlabel("Rounds", fontsize=20)
    plt.ylabel(r"Nash Regret", fontsize=20)
    plt.title(title, fontsize=20)           # change plot title based on dataset choosen 
    plt.legend(fontsize=20, columnspacing=0.1, handletextpad=0.1, labelspacing=0.1, borderpad=0.1 , framealpha=1, ncols = 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plots_dir}/FairLinUCB_vary_d_{dataset}.png")
    plt.close()

# **********************************************************************************************************************************




