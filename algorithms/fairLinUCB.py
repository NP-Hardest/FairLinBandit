import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linprog
import cvxpy as cp
from scipy.spatial import ConvexHull
import warnings
import time

warnings.filterwarnings("ignore")



class FairLinUCB:
    def __init__(self, X, T, p, sigma, d):
        self.arms = X
        self.num_arms = len(X)
        self.T = T
        self.p = p
        self.sigma = sigma
        self.d = d
        self.alpha_reg = 0.01
        self.all_indices = np.arange(self.num_arms)

        self.D_supp_idx, self.D_w = self.solve_d_optimal_design(self.all_indices)
        
        self.U_indices, self.U_weights = self.solve_john_ellipsoid_proxy(self.all_indices)
        

        self.total_rounds = 0
        self.history = []
        

        self.p_a = 1.0 if self.p >= -1 else self.p



    def solve_d_optimal_design(self, arm_indices):
        active_arms = self.arms[arm_indices] 
        N, d = active_arms.shape

        w = cp.Variable(N)


        weighted_X = cp.multiply(w[:, None], active_arms)
        U = active_arms.T @ weighted_X
        

        U += 1e-6 * np.eye(d)


        objective = cp.Maximize(cp.log_det(U))


        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]


        prob = cp.Problem(objective, constraints)
        prob.solve() 


        optimal_weights = w.value
        
        support_mask = optimal_weights > 1e-5
        return arm_indices[support_mask], optimal_weights[support_mask]
        

    def solve_john_ellipsoid_proxy(self, active_indices):
        """
        Approximate distribution U using the Chebyshev center
        of the convex hull of the active arms. If geometry fails, fallback to uniform over active_indices.
        """
        try:
            X_active = self.X[active_indices]
            if len(X_active) == 0:
                return None, None
            
        
            if len(X_active) <= self.d:
                alphas = np.ones(len(active_indices)) / len(active_indices)
                return alphas, active_indices

            hull = ConvexHull(X_active)
            hull_vertices = X_active[hull.vertices]
            A = hull.equations[:, :-1]
            b = -hull.equations[:, -1]

       
            c_obj = np.zeros(self.d + 1)
            c_obj[-1] = -1  
            norms = np.linalg.norm(A, axis=1, keepdims=True)
            A_ub = np.hstack([A, norms])
            b_ub = b

            res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub)
            if not res.success:
                alphas = np.ones(len(active_indices)) / len(active_indices)
                return alphas, active_indices
            center = res.x[:-1]


            Y = hull_vertices.T  
            c_feasibility = np.zeros(len(hull.vertices))
            A_eq = np.vstack([Y, np.ones(len(hull.vertices))])
            b_eq = np.hstack([center, 1])
            res_alpha = linprog(c=c_feasibility, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
            
            if not res_alpha.success:
                alphas = np.ones(len(active_indices)) / len(active_indices)
                return alphas, active_indices
            
            alphas = res_alpha.x
            

            alphas[alphas < 1e-9] = 0.0
            

            total_mass = alphas.sum()
            if total_mass < 1e-12:

                alphas = np.ones(len(alphas)) / len(alphas)
            else:

                alphas /= total_mass
                alphas[-1] = 1.0 - alphas[:-1].sum()
                alphas = np.clip(alphas, 0.0, 1.0) 
                alphas /= alphas.sum() 

            global_indices = np.array(active_indices)[hull.vertices]
            return global_indices, alphas
            
        except Exception:
            alphas = np.ones(len(active_indices)) / len(active_indices)
            return active_indices, alphas


    def pull_arms_subroutine(self, env, U_indices, U_weights, D_support_indices, D_weights, T_tilde, initial_V, initial_s):
        """
        Algorithm 1: PullArms 
        """

        V_curr = initial_V.copy()
        s_curr = initial_s.copy()
        

        active_D_indices = list(range(len(D_support_indices))) 
        D_counts = np.zeros(len(D_support_indices), dtype=int)
        
     
        D_targets = np.ceil(D_weights * T_tilde / 3.0).astype(int)
        
        rr_pointer = 0 
        
    
        for _ in range(int(T_tilde)):
            if self.total_rounds >= self.T: break
            
        
            flag = "U" if np.random.rand() < 0.5 else "D"
            
            selected_global_idx = -1
            
        
            if flag == "U" or len(active_D_indices) == 0:

                selected_global_idx = np.random.choice(U_indices, p=U_weights)
                
            elif flag == "D":
       
                if rr_pointer >= len(active_D_indices):
                    rr_pointer = 0
                
  
                current_ptr = active_D_indices[rr_pointer]
                selected_global_idx = D_support_indices[current_ptr]
                
 
                D_counts[current_ptr] += 1
                
     
                if D_counts[current_ptr] >= D_targets[current_ptr]:
                    active_D_indices.pop(rr_pointer)
             
                else:
                    rr_pointer += 1
            

            r = env.get_reward(selected_global_idx)
            self.history.append(selected_global_idx)
            self.total_rounds += 1
            
   
            x = self.arms[selected_global_idx]
            V_curr += np.outer(x, x)
            s_curr += r * x
            

        try:
            theta_hat = np.linalg.pinv(V_curr) @ s_curr
        except:
            theta_hat = np.zeros(self.d)
            

        return theta_hat, s_curr, V_curr

    def run(self, env):
   
        t_val = 1
        V = np.zeros((self.d, self.d))
        s = np.zeros(self.d)
        theta_hat = np.zeros(self.d)

    
        T_tilde = 36 * np.log(self.T)


        while True:
            preds = self.arms @ theta_hat
            conf_width = 4 * self.sigma * self.d * np.sqrt((3 * np.log(self.T)) / t_val)
            

            lhs_inner = np.max(preds) - conf_width
            
            rhs_num = (900 * (self.p_a**2) * (self.sigma**2) * (self.d**2) * np.log(self.T)) / t_val
            
            if lhs_inner <= 0 or lhs_inner <= (rhs_num / lhs_inner):
       
                theta_hat, s, V = self.pull_arms_subroutine(
                    env, self.U_indices, self.U_weights, self.D_supp_idx, self.D_w, T_tilde, V, s
                )
                t_val += int(T_tilde)
                T_tilde *= 2
                if self.total_rounds >= self.T: break
            else:
                break
        print("Exited Phase I at total rounds: ", self.total_rounds)

        tau = T_tilde / 2
        V_t = V + self.alpha_reg * np.eye(self.d)
        s_t = s


        for t in range(int(tau) + 1, self.T + 1):
            if self.total_rounds >= self.T: break
            

            V_inv = np.linalg.pinv(V_t)
            theta_t = V_inv @ s_t
            

            term1 = self.d * np.log(1 + t / (self.d * self.alpha_reg))
            beta_t = self.sigma * np.sqrt(term1 + 2 * np.log(self.T)) + np.sqrt(self.alpha_reg)
            
  
            # ucb_values = []
            # for i in range(self.num_arms):
            #     x_i = self.arms[i]
            #     norm_val = np.sqrt(x_i.T @ V_inv @ x_i)
            #     ucb_values.append(np.dot(x_i, theta_t) + beta_t * norm_val)
            
            # x_t_idx = np.argmax(ucb_values)
            
   
            # r_t = env.get_reward(x_t_idx)
            # x_t = self.arms[x_t_idx]

  
            mean_rewards = self.arms @ theta_t


            variances = np.sum((self.arms @ V_inv) * self.arms, axis=1)
            norms = np.sqrt(np.maximum(variances, 0)) 

            ucb_values = mean_rewards + beta_t * norms
            x_t_idx = np.argmax(ucb_values)

 
            r_t = env.get_reward(x_t_idx)
            x_t = self.arms[x_t_idx]
            
            V_t += np.outer(x_t, x_t)
            s_t += r_t * x_t
            
            self.history.append(x_t_idx)
            self.total_rounds += 1

            if(self.total_rounds%1000000==0):
                print("Total Rounds: ", self.total_rounds)

        return np.array(self.history)


def simulate_fairLinUCB(env, X, T, num_trials=10, sigma2=1.0, p=0):
    mu_star = np.max(env.mean_rewards)
    total_rewards = []
    
    for _ in tqdm(range(num_trials), desc="FairLinUCB Trials"):
        algo = FairLinUCB(X, T, p=0.0, sigma=np.sqrt(sigma2), d=X.shape[1])
        history = algo.run(env)
        
        if len(history) < T:
            history = np.concatenate([history, np.full(T - len(history), history[-1])])
        else:
            history = history[:T]
            
        total_rewards.append(env.mean_rewards[history])


    for i in range(len(total_rewards)-1):    
        if(np.array_equal(total_rewards[i],total_rewards[i+1])):
            print("Equal")
        else:
            print("Not Equal")

    expected_means = np.mean(total_rewards, axis=0)
    
    # cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-10)))
    # inv_t = 1.0 / np.arange(1, T+1)
    # geom_mean = np.exp(cumsum_log * inv_t)
    
    # return mu_star - geom_mean

    if p == 0:  
        cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-300)))
        inv_t = 1.0 / np.arange(1, T+1)
        geom_mean = np.exp(cumsum_log * inv_t)
        avg_regret = mu_star - geom_mean
    elif p == 1:  
        cum_rewards = np.cumsum(expected_means)
        inv_t = 1.0 / np.arange(1, T+1)
        arith_mean = cum_rewards * inv_t
        avg_regret = mu_star - arith_mean
    else:  
        p_powers = np.power(expected_means, p)              
        # print(p_powers)
        cum_p_powers = np.cumsum(p_powers)
        inv_t = 1.0 / np.arange(1, T + 1)
        p_mean = np.power(cum_p_powers * inv_t, 1.0 / p)
        avg_regret = mu_star - p_mean
        
    # return avg_regret

        
        
    return avg_regret

