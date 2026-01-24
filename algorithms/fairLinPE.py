import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linprog
import cvxpy as cp
from scipy.spatial import ConvexHull
import warnings


warnings.filterwarnings("ignore")


class FairLinPE:
    def __init__(self, X, T, p, sigma, d):
        self.arms = X
        self.num_arms = len(X)
        self.T = T
        self.p = p
        self.sigma = sigma
        self.d = d
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

                # local_u = np.random.choice(len(U_indices))
                # selected_global_idx = U_indices[local_u]
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


            # try:
            V_curr_inv = np.linalg.pinv(V_curr)

            # except:
            #     print(f"V not invertible for T_tilde = {T_tilde} at t = {_+1}")    

            s_curr += r * x

        try:
            V_curr_inv = np.linalg.inv(V_curr)
            # print("Minimum eigen value at end of phase 1: ", np.min(np.linalg.eigvals(V_curr_inv)))    
            theta_hat = V_curr_inv @ s_curr
        except:
            theta_hat = np.zeros(self.d)
            

        return theta_hat, s_curr, V_curr

    def run(self, env):

       
        theta_curr = np.zeros(self.d)

        V_curr = np.zeros((self.d, self.d))
        s_curr = np.zeros(self.d)
        

        T_tilde = 36 * np.log(self.T)
        t_phase1 = 0
        

        while self.total_rounds < self.T:
            
            condition_holds_for_all = True
            

            if t_phase1 > 1:
                # print("here")
                denom_offset = 4.0 * np.sqrt(3 * (self.d**2) * (self.sigma**2) * np.log(self.T) / t_phase1)
                

                preds = self.arms @ theta_curr
                

                best_idx = np.argmax(preds)
                pred_max = preds[best_idx]

                denom = pred_max - denom_offset
                

                if denom > 1e-9:

                    lhs = (t_phase1 * pred_max) / (3.0 * self.d)
                    

                    term1 = (300.0 * (self.p_a**2) * self.d * (self.sigma**2) * np.log(self.T)) / denom
                    term2 = (4.0/3.0) * np.sqrt(3 * t_phase1 * (self.sigma**2) * np.log(self.T))
                    rhs = term1 + term2

                    if lhs > rhs:
                        condition_holds_for_all = False
        

            if t_phase1 <= 1 or condition_holds_for_all:

                theta_curr, s_curr, V_curr = self.pull_arms_subroutine(env, self.U_indices, self.U_weights, self.D_supp_idx, self.D_w, T_tilde, V_curr, s_curr)

                t_phase1 += int(T_tilde)
                T_tilde *= 2
            else:
                break
        

        T_tilde = max(1, T_tilde / 2.0)
        

        preds = self.arms @ theta_curr
        gamma = np.max(preds)
        

        width = 8.0 * np.sqrt((self.d**2 * self.sigma**2 * np.log(self.T)) / T_tilde)
        survivors = np.where(preds >= (gamma - width))[0]
        

        T_prime = (2.0/3.0) * T_tilde
        print(f"Part I End: {len(survivors)} arms surviving.")


        print(self.total_rounds)
        while self.total_rounds < self.T:
            

            V_ep = np.zeros((self.d, self.d))
            s_ep = np.zeros(self.d)
            

            D_supp_idx_II, D_w_II = self.solve_d_optimal_design(survivors)
            

            for i, global_idx in enumerate(D_supp_idx_II):
                weight = D_w_II[i]

                N_a = int(np.ceil(weight * T_prime))
                
                arm_vec = self.arms[global_idx]
                for _ in range(N_a):
                    if self.total_rounds >= self.T: break
                    r = env.get_reward(global_idx)
                    self.history.append(global_idx)
                    self.total_rounds += 1
                    

                    V_ep += np.outer(arm_vec, arm_vec)
                    s_ep += r * arm_vec
                    

            try:
                V_ep_inv = np.linalg.inv(V_ep)
                # print(f"minimum eigen value: {np.min(np.linalg.eigvals(V_ep_inv))}")    
                theta_hat = V_ep_inv @ s_ep
            except:
                theta_hat = np.zeros(self.d)
            


            if len(survivors) > 0:
                preds_survivors = self.arms[survivors] @ theta_hat
                gamma = np.max(preds_survivors)
            
                width = 8.0 * np.sqrt((self.d**2 * self.sigma**2 * np.log(self.T)) / T_prime)
                threshold = gamma - width
                
                mask = preds_survivors >= threshold
                survivors = survivors[mask]
            

            T_prime *= 2
            print(f"Episode End: {len(survivors)} arms surviving.")
        return np.array(self.history)



def simulate_fairLinPE(env, X, T, num_trials=10, sigma2=1.0, p =0):
    mu_star = np.max(env.mean_rewards)
    total_rewards = []
    
    for _ in tqdm(range(num_trials), desc="FairLinPE Trials"):
        algo = FairLinPE(X, T, p=0.0, sigma=np.sqrt(sigma2), d=X.shape[1])
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
