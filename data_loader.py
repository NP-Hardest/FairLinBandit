import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.decomposition import TruncatedSVD  


def get_max_arms(qids):
    _, counts = np.unique(qids, return_counts=True)
    return np.max(counts)



# for generating synthetic data with number of arms = n_arms
def generate_synthetic_data_with_lasso(n_arms=1000, d_dim=10, n_train_samples=2000, seed=42):
    """
    1. Generates a 'true' hidden theta.
    2. Generates a training set to 'learn' theta_star via Lasso.
    3. Generates the actual experimental arms.
    """
    rng = np.random.default_rng(seed)
    

    hidden_theta = np.zeros(d_dim)
    hidden_theta[:d_dim//2] = rng.uniform(0.5, 1.0, d_dim//2) 
    

    X_train = rng.standard_normal((n_train_samples, d_dim))

    y_train = X_train @ hidden_theta + rng.normal(0, 0.1, n_train_samples)
    

    model = Lasso(alpha=0.001, fit_intercept=False, max_iter=10000)
    model.fit(X_train, y_train)
    theta_star = model.coef_
    

    norm_theta = np.linalg.norm(theta_star)
    if norm_theta > 0:
        theta_star /= norm_theta


    X_raw = rng.standard_normal((n_arms, d_dim))
    

    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    X_arms = X_raw / (norms + 1e-9)

    count_flipped = 0
    for i in range(n_arms):
        if np.dot(X_arms[i], theta_star) < 1e-3:
            X_arms[i] *= -1
            count_flipped += 1
            
    print(f"Lasso recovered theta_star with {np.count_nonzero(theta_star)} non-zero coefficients.")
    print(f"Generated {n_arms} arms. Flipped {count_flipped} for positivity.")
    
    return X_arms, theta_star




def extract_and_load_yahoo(tar_path, fold='set1', split='train'):
    """
    Extracts Yahoo LTR data from .tar.bz2 and loads it.
    
    Args:
        tar_path: Path to the .tar.bz2 file
        fold: 'set1' (US) or 'set2' (Asia)
        split: 'train', 'valid', or 'test'
    """
    
    
    print(f"Opening {tar_path}...")
    
    target_file = None
    
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            if f"{fold}.{split}" in member.name and member.name.endswith(".txt"):
                print(f"Found {member.name}, extracting...")
                tar.extract(member, path=".") 
                target_file = member.name
                break
        
        if target_file is None:
            raise ValueError(f"Could not find {fold}.{split}.txt in {tar_path}")


    print(f"Loading {target_file} into memory...")
    X, y, qids = load_svmlight_file(target_file, query_id=True)
    
    print(f"Loaded {X.shape[0]} documents with {X.shape[1]} features.")




    
    MAX_ACTIONS = get_max_arms(qids)
    print(f"Max documents in any query: {MAX_ACTIONS}")

    return X, y, qids

def load_and_process_yahoo_tar(tar_path, d_dim=10, fold='set1', split='train'):
    
    
    # ********************************************** <start> Uncomment if running for the first time *******************************************************


    # print(f"Processing Yahoo! LTR from {tar_path}...")

    # target_filename = None
    
    # search_suffix = f"{fold}.{split}.txt"
    
    # print("Scanning archive for dataset file...")
    # with tarfile.open(tar_path, "r:bz2") as tar:
    #     for member in tar.getmembers():
    #         if member.name.endswith(search_suffix):
    #             print(f"Found {member.name}, extracting...")
    #             tar.extract(member, path=".")
    #             target_filename = member.name
    #             break
                
    # if target_filename is None:
    #     raise FileNotFoundError(f"Could not find *{search_suffix} in {tar_path}")


    # print("Loading data into memory (this may take a moment)...")
    # X_all, y_all, qids = load_svmlight_file(target_filename, query_id=True)
    
    # X_dense = X_all.toarray()
    

    # unique_qids, indices = np.unique(qids, return_index=True)
    # _, counts = np.unique(qids, return_counts=True)
    # MAX_ACTIONS = np.max(counts)


    # ********************************************** <end> Uncomment if running for the first time *******************************************************

    
    
    
    
    MAX_ACTIONS = 139
    NUM_FEATURES = 699
    
    
    
    
    
    # ********************************************** <start> Uncomment if running for the first time *******************************************************


    # print(f"Dataset Stats: {len(unique_qids)} queries, {NUM_FEATURES} features.")
    # print(f"Max documents per query (MAX_ACTIONS): {MAX_ACTIONS}")

    # indices = np.append(indices, len(qids))


    # print("Averaging features across queries to create fixed arms...")
    
    # feature_sums = np.zeros((MAX_ACTIONS, NUM_FEATURES))
    # reward_sums = np.zeros(MAX_ACTIONS)
    # counts_per_pos = np.zeros(MAX_ACTIONS)


    # for i in range(len(unique_qids)):
    #     start_idx = indices[i]
    #     end_idx = indices[i+1]
        
    #     block_X = X_dense[start_idx:end_idx]
    #     block_y = y_all[start_idx:end_idx]
        
    #     num_docs = block_X.shape[0]
        

    #     limit = min(num_docs, MAX_ACTIONS)
    #     feature_sums[:limit] += block_X[:limit]
    #     reward_sums[:limit] += block_y[:limit]
    #     counts_per_pos[:limit] += 1


    # counts_per_pos[counts_per_pos == 0] = 1 
    
    # X_avg = feature_sums / counts_per_pos[:, None]
    # y_avg = reward_sums / counts_per_pos


    # # os.remove(target_filename) 
    # np.savez("yahoo_pre_pca.npz", features=X_avg, rewards=y_avg)
    # print("Pre-PCA dataset saved to 'yahoo_pre_pca.npz'.")




    # ********************************************** <end> Uncomment if running for the first time *******************************************************


    data = np.load("yahoo_pre_pca.npz")


    X_avg = data['features']
    y_avg = data['rewards']


    print(f"Reducing dimension from {NUM_FEATURES} to {d_dim}...")
    svd = TruncatedSVD(n_components=d_dim, random_state=42)
    X_reduced = svd.fit_transform(X_avg)


    print("Training Lasso on averaged data to define Theta*...")
    model = Lasso(alpha=0.001, fit_intercept=False, max_iter=10000)
    model.fit(X_reduced, y_avg)
    
    theta_star = model.coef_
    norm_theta = np.linalg.norm(theta_star)
    if norm_theta > 0:
        theta_star /= norm_theta


    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    X_arms = X_reduced / (norms + 1e-9)

    count_flipped = 0
    for i in range(MAX_ACTIONS):
        if np.dot(X_arms[i], theta_star) < 1e-6:
            X_arms[i] *= -1
            count_flipped += 1
            
    print(f"Flipped {count_flipped} arms to ensure positive means.")
    
    return X_arms, theta_star








def load_and_process_mslr(filepath, d_dim=10):


# ********************************************** <start> Uncomment if running for the first time *******************************************************



#     print("Loading MSLR data with Query IDs...")
#   
#     X_all, y_all, qids = load_svmlight_file(filepath, query_id=True)

#     MAX_ACTIONS = get_max_arms(qids)
#     print(f"Max documents in any query: {MAX_ACTIONS}")
    
#    
#   
#     X_combined = X_all[:, :135]
#     print(f"Kept Title+Body features. Shape: {X_combined.shape}")

#  
#     print("Averaging features across queries to create fixed arms...")
    
#    
#     unique_qids, indices = np.unique(qids, return_index=True)
#     indices = np.append(indices, len(qids))

# ********************************************** <end> Uncomment if running for the first time *******************************************************


  
  
    MAX_ACTIONS = 908 


    
# ********************************************** <start> Uncomment if running for the first time *******************************************************




#     NUM_FEATURES = 135
    


#     feature_sums = np.zeros((MAX_ACTIONS, NUM_FEATURES))
#     reward_sums = np.zeros(MAX_ACTIONS)
#     counts = np.zeros(MAX_ACTIONS)


#     for i in range(len(unique_qids)):
#         start_idx = indices[i]
#         end_idx = indices[i+1]
        


#         block_X = X_combined[start_idx:end_idx].toarray()
#         block_y = y_all[start_idx:end_idx]
        
#         num_docs = block_X.shape[0]
        


#         limit = min(num_docs, MAX_ACTIONS)
#         feature_sums[:limit] += block_X[:limit]
#         reward_sums[:limit] += block_y[:limit]
#         counts[:limit] += 1

#     counts[counts == 0] = 1 
    
#     X_avg = feature_sums / counts[:, None]
#     y_avg = reward_sums / counts
    
    
#     print("Pre-PCA dataset saved to 'mslr_pre_pca.npz'.")

# ********************************************** <end> Uncomment if running for the first time *******************************************************


    data = np.load("mslr_pre_pca.npz")


    X_avg = data['features']
    y_avg = data['rewards']

    print(f"Loaded features shape: {X_avg.shape}") 

    print(f"Created {X_avg.shape[0]} fixed arms (averaged actions).")


    print(f"Reducing dimension from 135 to {d_dim}...")
    svd = TruncatedSVD(n_components=d_dim, random_state=42)
    X_reduced = svd.fit_transform(X_avg)


    print("Training Lasso on averaged data to define Theta*...")
    model = Lasso(alpha=0.001, fit_intercept=False, max_iter=10000)
    model.fit(X_reduced, y_avg)
    
    theta_star = model.coef_

    norm_theta = np.linalg.norm(theta_star)
    if norm_theta > 0:
        theta_star /= norm_theta


    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    X_arms = X_reduced / (norms + 1e-9)


    count_flipped = 0
    for i in range(MAX_ACTIONS):
        if np.dot(X_arms[i], theta_star) < 1e-6:
            X_arms[i] *= -1
            count_flipped += 1
            
    print(f"Flipped {count_flipped} arms to ensure positive means.")
    
    return X_arms, theta_star


