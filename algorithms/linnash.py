import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linprog
import cvxpy as cp
from scipy.spatial import ConvexHull
import warnings

warnings.filterwarnings("ignore")



class LinNash:
    def __init__(self, X, T, sigma, d):
        self.arms = X
        self.num_arms = len(X)
        self.T = T
        self.sigma = sigma 
        self.nu = sigma**2 
        self.d = d
        self.all_indices = np.arange(self.num_arms)
        self.D_idx, self.D_weights = self.solve_d_optimal_design(self.all_indices)
        self.U_idx, self.U_weights = self.solve_john_ellipsoid_proxy(self.all_indices)
        

        self.total_rounds = 0
        self.history = []
        
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


    def generate_arm_sequence(self, env, current_arm_indices, T_tilde, initial_V):
        """
        Algorithm 1: GenerateArmSequence
        Executes the sequence generation and pulling simultaneously.
        """
        active_D_indices = list(range(len(self.D_idx))) # Indices pointing to D_idx
        c_z = np.zeros(len(self.D_idx), dtype=int)
        
        D_targets = np.ceil(self.D_weights * T_tilde / 3.0).astype(int)
        

        # U_idx, U_weights = self.solve_john_ellipsoid_proxy(current_arm_indices)
        
        V_curr = initial_V.copy()
        s_curr = np.zeros(self.d) 
        
        rr_pointer = 0 
        

        for _ in range(int(T_tilde)):
            if self.total_rounds >= self.T: break
            
            flag = "SAMPLE-U" if np.random.rand() < 0.5 else "D/G-OPT"
            
            selected_global_idx = -1
            
            if flag == "SAMPLE-U" or len(active_D_indices) == 0:
                selected_global_idx = np.random.choice(self.U_idx, p=self.U_weights)
                
            elif flag == "D/G-OPT":
                if rr_pointer >= len(active_D_indices):
                    rr_pointer = 0
                
                current_ptr_idx = active_D_indices[rr_pointer]
                selected_global_idx = self.D_idx[current_ptr_idx]
                
                c_z[current_ptr_idx] += 1
                
                if c_z[current_ptr_idx] >= D_targets[current_ptr_idx]:
                    active_D_indices.pop(rr_pointer)
                else:
                    rr_pointer += 1
            
            r = env.get_reward(selected_global_idx)
            self.history.append(selected_global_idx)
            self.total_rounds += 1
            
            x = self.arms[selected_global_idx]
            V_curr += np.outer(x, x)
            s_curr += r * x
            
        return V_curr, s_curr

    def run(self, env):
        
        V = np.zeros((self.d, self.d))
        term_inside = self.T * (self.d ** 2.5) * self.nu * np.log(self.T)
        T_tilde = 3 * np.sqrt(term_inside)
        
        print(f"Starting Part I with T_tilde = {int(T_tilde)}")
        
        survivors = np.arange(self.num_arms)
        V, s_sum = self.generate_arm_sequence(env, survivors, T_tilde, V)
        
        try:
            theta_hat = np.linalg.pinv(V) @ s_sum
        except:
            theta_hat = np.zeros(self.d)
            
        preds = self.arms @ theta_hat
        gamma = np.max(preds)
        
        gamma_safe = max(1e-9, gamma) 
        numerator = 3 * gamma_safe * (self.d ** 2.5) * self.nu * np.log(self.T)
        width = 16 * np.sqrt(numerator / T_tilde)
        
        surviving_indices = np.where(preds >= (gamma - width))[0]
        survivors = survivors[surviving_indices]
        
        T_prime = (2.0 / 3.0) * T_tilde
        
        print(f"Part I End: {len(survivors)} arms surviving.")
        while self.total_rounds < self.T:
            
            V_phase = np.zeros((self.d, self.d))
            s_phase = np.zeros(self.d)
            
            D_idx, D_weights = self.solve_d_optimal_design(survivors)
            
            for i, global_idx in enumerate(D_idx):
                weight = D_weights[i]
                
                count = int(np.ceil(weight * T_prime))
                
                arm_vec = self.arms[global_idx]
                
                for _ in range(count):
                    if self.total_rounds >= self.T: break
                    r = env.get_reward(global_idx)
                    self.history.append(global_idx)
                    self.total_rounds += 1
                    
                    V_phase += np.outer(arm_vec, arm_vec)
                    s_phase += r * arm_vec
            
            if self.total_rounds >= self.T: break
            
            try:
                theta_hat = np.linalg.pinv(V_phase) @ s_phase
            except:
                theta_hat = np.zeros(self.d)
                
            if len(survivors) > 0:
                preds_survivors = self.arms[survivors] @ theta_hat
                gamma = np.max(preds_survivors)
                gamma_safe = max(1e-9, gamma)
                numerator = gamma_safe * (self.d ** 2.5) * np.log(self.T)
                numerator *= self.nu 
                
                width = 16 * np.sqrt(numerator / T_prime)
                
                preds_survivors = self.arms[survivors] @ theta_hat
                mask = preds_survivors >= (gamma - width)
                survivors = survivors[mask]
            

            print(f"Episode End: {len(survivors)} arms surviving.")
            T_prime *= 2.0
            
        return np.array(self.history)


def simulate_linnash(env, X, T, num_trials=10, sigma2=1.0):
    mu_star = np.max(env.mean_rewards)
    total_rewards = []
    
    for _ in tqdm(range(num_trials), desc="LinNash Trials"):
        algo = LinNash(X, T, sigma=np.sqrt(sigma2), d=X.shape[1])
        history = algo.run(env)
        
        if len(history) < T:
            history = np.concatenate([history, np.full(T - len(history), history[-1])])
        else:
            history = history[:T]
            
        total_rewards.append(env.mean_rewards[history])
        
    expected_means = np.mean(total_rewards, axis=0)
    
    cumsum_log = np.cumsum(np.log(np.maximum(expected_means, 1e-10)))
    inv_t = 1.0 / np.arange(1, T+1)
    geom_mean = np.exp(cumsum_log * inv_t)
    
    return mu_star - geom_mean

