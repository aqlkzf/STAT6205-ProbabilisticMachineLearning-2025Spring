# %% [markdown]
# 
# # Probabilistic Principal Component Analysis (PPCA) on Digits Dataset
# 
# This code implements PPCA by two methods:
# 1. ML solution using spectral decomposition.
# 2. EM algorithm for PPCA.
# 
# The data is loaded from the digits dataset in sklearn, and we perform dimensionality reduction and projection on the training set.
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, inv
from sklearn.datasets import load_digits
import matplotlib.animation as animation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# Set display configuration
%config InlineBackend.figure_format = "retina"
np.set_printoptions(precision=4, suppress=True)


# %% [markdown]
# 
# ## Data Loading
# Load the digits dataset.

# %%
digits = load_digits()
X = digits.data      # Each sample is 64-dimensional (8x8 image)
y = digits.target
print("X shape:", X.shape)
print("Unique labels:", np.unique(y))

Xtrain = X
ytrain = y
N, D = Xtrain.shape
print("Train set shape:", Xtrain.shape)

# %%
# This function is used to visualize the digits No need to fully understand it
# I just want to show you the data
def visualize_digits(X, y, n_samples=10, random_state=42):
    """
    Visualize random digit samples from the dataset
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, 64)
        The digit image data (flattened 8x8 images)
    y : array-like of shape (n_samples,)
        The labels
    n_samples : int
        Number of samples to visualize
    random_state : int
        Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Randomly select indices
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    # Create subplot grid
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    
    # Flatten axes for easier iteration
    axes = axes.ravel()
    
    # Plot each digit
    for idx, ax in enumerate(axes):
        if idx < n_samples:
            # Reshape the flattened image back to 8x8
            digit_image = X[indices[idx]].reshape(8, 8)
            
            # Plot the image
            ax.imshow(digit_image, cmap='gray')
            ax.set_title(f'Label: {y[indices[idx]]}')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()




# %% [markdown]
# We visualize 15 samples from X and corresponding label y.

# %%
visualize_digits(X, y, n_samples=15)

# %% [markdown]
# 
# ## ML PPCA
# Compute ML estimates using spectral decomposition and project the data to a lower-dimensional space.
# 
# Formulas:
# - Mean:  $\mu = \bar{x}$
# - Noise variance:  $\sigma^2_{ML} = \frac{1}{D-M} \sum_{m=M+1}^{D} \lambda_m$
# - Loading matrix:  $W_{ML} = U_M \left(L_M - \sigma^2_{ML} I\right)^{1/2}$
# - Posterior mean:  $E[z] = (W^T W+\sigma^2 I)^{-1} W^T (x-\mu)$
# - Where:
#   *  ${\bf U}_M$ is an $D \times M$ matrix whose columns are given by the $M$ eigenvectors with corresponding largest $M$ eigenvalues
#   * ${\bf L}_M$ is a diagonal $M\times M$ matrix of the corresponding eigenvectors
#   * ${\bf R}$ is an arbitrary orthogonal $M\times M$ matrix( we take ${\bf R}={\bf I}$)
# 

# %%
# Set latent dimension
M = 2

# %%

# Compute mean and center the data (using training set)
mu = Xtrain.mean(axis=0)
X_centered = Xtrain - mu


# %%

# Compute covariance matrix
S = np.cov(X_centered.T)


# %%

# Eigen-decomposition and sort in descending order
eigenvalues, eigenvectors = eigh(S)
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]


# %%

# Compute ML noise variance using the remaining eigenvalues
sigma2_ml = eigenvalues[M:].sum() / (D - M)


# %%

# Compute W_ML (using R as identity)
L_M = np.diag(eigenvalues[:M])
W_ml = eigenvectors[:, :M] @ np.sqrt(L_M - sigma2_ml * np.eye(M))


# %%

# Compute posterior projection: E[z] = (W^TW+sigma^2 I)^{-1} W^T (X-mu)
M_mat = W_ml.T @ W_ml + sigma2_ml * np.eye(M)
Ez_ml = np.linalg.inv(M_mat) @ W_ml.T @ X_centered.T  # shape: (M, N)


# %%

# Plot the ML PPCA projection result
plt.figure(figsize=(8,6))
plt.scatter(Ez_ml[0, :], Ez_ml[1, :], c=ytrain, cmap="tab10", s=10)
plt.title("ML PPCA Projection on Digits (Train Set)")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.colorbar()
plt.show()

# %% [markdown]
# ### Put all things together into a class

# %%
class MLPPCA:
    def __init__(self, n_components):
        """
        Initialize ML-PPCA
        
        Parameters:
        -----------
        n_components : int
            Number of components to keep (M in the equations)
        """
        self.M = n_components
        self.W = None  # Weight matrix
        self.mu = None # Mean vector
        self.sigma2 = None # Noise variance
        
    def fit(self, X):
        """
        Fit the ML-PPCA model
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        """
        # Center the data
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu
        
        # Compute covariance matrix
        N = X.shape[0]
        # S = np.dot(X_centered.T, X_centered) / N
        S = np.cov(X.T)
        # Compute eigendecomposition
        eigenvals, eigenvecs = eigh(S)
        # Sort in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Compute ML solution for sigma^2
        self.sigma2 = np.mean(eigenvals[self.M:])
        
        # Compute W matrix
        # Take top M eigenvectors
        U_M = eigenvecs[:, :self.M]
        L_M = np.diag(eigenvals[:self.M])
        
        # R can be identity matrix as it's arbitrary orthogonal
        R = np.eye(self.M)
        
        # Compute W
        self.W = U_M @ np.sqrt(L_M - self.sigma2 * np.eye(self.M)) @ R
        
        return self
    
    def transform(self, X):
        """
        Transform data to latent space
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        Z : array-like of shape (n_samples, n_components)
            Transformed data in latent space
        """
        X_centered = X - self.mu
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.M)
        Z = X_centered @ self.W @ np.linalg.inv(M)
        return Z
    
    def fit_transform(self, X):
        """
        Fit the model and transform the data
        """
        return self.fit(X).transform(X)

# %%
# Example usage
ml_ppca = MLPPCA(n_components=2)
Z = ml_ppca.fit_transform(X)

# visualize the result
plt.figure(figsize=(8,6))
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10', s=10)
plt.title('ML-PPCA Projection')
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.colorbar()
plt.show()

# %%


# %% [markdown]
# 
# ## EM Algorithm Implementation for PPCA
# 
# The EM algorithm for PPCA works as follows:
# - **E-step:** For each sample $x_n$, compute:
#   $$
#   E[z_n] = (W^T W+\sigma^2 I)^{-1}W^T(x_n-\bar{x})
#   $$
#   $$
#   E[z_n z_n^T] = E[z_n]E[z_n]^T + \sigma^2 (W^T W+\sigma^2 I)^{-1}
#   $$
# - **M-step:** Update parameters:
#   $$
#   W_{\text{new}} = \left(\sum_{n}(x_n-\bar{x})\,E[z_n]^T\right)\left(\sum_{n}E[z_n z_n^T]\right)^{-1}
#   $$
#   $$
#   \sigma^2_{\text{new}} = \frac{1}{ND}\sum_{n}\left(\|x_n-\bar{x}\|^2 - 2\,E[z_n]^T\,W^T(x_n-\bar{x}) + \operatorname{Tr}(W^TW\,E[z_n z_n^T])\right)
#   $$
# 

# %%

class ProbabilisticPCA:
    """
    EM algorithm implementation for PPCA.
    Model: x = Wz + mu + eps, with eps ~ N(0, sigma2 I) and z ~ N(0, I)
    """
    def __init__(self, X, M, W_init=None, sigma2_init=None, seed=None):
        self.seed = seed
        self.M = M
        self.N, self.D = X.shape
        self.Xbar = X - X.mean(axis=0)
        self.Xnorm2 = np.sum(self.Xbar**2)
        self.W = self._initialize_W(W_init)
        self.sigma2 = self._initialize_sigma2(sigma2_init)
        self.E_zn, self.E_znzn = self._compute_expectations()
        
    def _initialize_W(self, W_init):
        np.random.seed(self.seed)
        if W_init is None:
            return np.random.randn(self.D, self.M)
        else:
            return W_init
        
    def _initialize_sigma2(self, sigma2_init):
        np.random.seed(self.seed)
        if sigma2_init is None:
            return np.random.rand()
        else:
            return sigma2_init
        
    def _compute_expectations(self):
        # E-step: Compute M = W^T W + sigma2 I and its inverse.
        Mz = self.W.T @ self.W + self.sigma2 * np.eye(self.M)
        i_Mz = inv(Mz)
        # Compute E[z] = M^{-1} W^T (X - mu)^T, shape: (M, N)
        E_zn = np.linalg.inv(Mz) @ (self.W.T @ self.Xbar.T)
        # Compute E[z z^T] for each sample: outer(E[z_n]) + sigma2 * inv(M)
        E_znzn = np.empty((self.M, self.M, self.N))
        for n in range(self.N):
            ez = E_zn[:, n][:, None]
            E_znzn[:, :, n] = ez @ ez.T + self.sigma2 * i_Mz
        return E_zn, E_znzn
    
    def update_W(self):
        # Compute T1 = sum_{n}(x_n - mu) E[z_n]^T.
        # Xbar has shape (N, D) and E[z_n]^T has shape (N, M).
        T1 = self.Xbar.T @ self.E_zn.T  # shape: (D, M)
        # Compute T2 = sum_{n} E[z_n z_n^T]
        T2 = np.sum(self.E_znzn, axis=2)  # shape: (M, M)
        W_new = T1 @ inv(T2)
        return W_new

    def update_sigma2(self):
        total = 0
        # Sum over all samples:
        for n in range(self.N):
            x_n = self.Xbar[n]
            ez = self.E_zn[:, n]
            Ezz = self.E_znzn[:, :, n]
            total += np.linalg.norm(x_n)**2 - 2 * ez.T @ (self.W.T @ x_n) + np.trace(self.W.T @ self.W @ Ezz)
        sigma2_new = total / (self.N * self.D)
        return sigma2_new   
    
    def _update_parameters(self):
        # One EM update: first update sigma2, then update W.
        sigma2_new = self.update_sigma2()
        W_new = self.update_W()
        return sigma2_new, W_new
        
    def data_log_likelihood(self):
        # Compute the complete data log-likelihood Q value.
        # Here, Q is defined as the negative log-likelihood.
        Q = ( - self.N * self.M * np.log(2*np.pi)/2
              - np.sum([np.trace(self.E_znzn[:, :, n]) for n in range(self.N)])
              - self.D * np.log(2*np.pi*self.sigma2)/2
              - self.Xnorm2/(2*self.sigma2)
              + np.sum([ (self.W.T @ self.Xbar[n][:,None]).T @ self.E_zn[:, n][:,None] 
                         for n in range(self.N) ])/self.sigma2
              - np.sum([np.trace(self.W.T @ self.W @ self.E_znzn[:, :, n]) 
                         for n in range(self.N)])/self.sigma2 )
        return Q
    
    def project(self):
        # Compute projection: Xproj = (W^TW+sigma2 I)^{-1} W^T (X - mu)^T.
        Mz = self.W.T @ self.W + self.sigma2 * np.eye(self.M)
        Xproj = inv(Mz) @ self.W.T @ self.Xbar.T
        return Xproj
    
    def EM_step(self):
        # Perform one EM step: first E-step, then update parameters, then compute Q.
        self.E_zn, self.E_znzn = self._compute_expectations()
        self.sigma2, self.W = self._update_parameters()
        Q = -self.data_log_likelihood()
        return Q


# %% [markdown]
# 
# ## EM Algorithm Iterations and Results
# Run multiple EM iterations, record the Q values and projection results.
# Q is defined as the complete data log-likelihood. If Q increases over iterations, the algorithm converges.
# 

# %%

# Run EM algorithm using the training set.
M_em = 2  # latent dimension
ppca_em = ProbabilisticPCA(Xtrain, M_em, sigma2_init=10, seed=2718)

n_iter = 30
Qhist = [ppca_em.data_log_likelihood()]
proj_list = [ppca_em.project()]
for i in range(n_iter):
    Q = ppca_em.EM_step()
    Qhist.append(Q)
    proj_list.append(ppca_em.project())
    # Stop if Q converges (adjust tolerance as needed)
    if i > 0 and (Qhist[-1] - Qhist[-2] < 1e-3):
        break

# Plot the evolution of the log-likelihood
plt.figure(figsize=(8,6))
plt.plot(Qhist, marker='o')
plt.title("EM PPCA: Log-Likelihood Evolution on Digits (Train Set)")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.show()

# Plot the final EM projection result
proj_em = ppca_em.project()
plt.figure(figsize=(8,6))
plt.scatter(proj_em[0, :], proj_em[1, :], c=ytrain, cmap="tab10", s=10)
plt.title("EM PPCA Projection on Digits (Train Set)")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.colorbar()
plt.show()

# Optionally, create an animation of the projection evolution
fig, ax = plt.subplots(figsize=(8,6))
def animate(i):
    ax.cla()
    proj = proj_list[i]
    ax.scatter(proj[0, :], proj[1, :], c=ytrain, cmap="tab10", s=10)
    ax.set_title(f"Iteration {i}")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.axis("off")
ani = animation.FuncAnimation(fig, animate, frames=len(proj_list), interval=500)
ani.save("ppca_digits.gif", writer="imagemagick")
plt.show()

# %% [markdown]
# ## Task4

# %% [markdown]
# The following code show example how to using standard PCA extract the latent embeddings and project the latent embeddings using t-SNE.

# %%
import time
from sklearn.decomposition import PCA

def run_pca_once(X, n_comp):
    """
    Run PCA once and return running time and results
    """
    start_time = time.time()
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(X)
    end_time = time.time()
    return end_time - start_time, pca, Z

def compare_svd_pca_components(X, y, components_list=[2, 5, 10, 20], n_runs=3):
    """
    Compare PCA results with different numbers of components using SVD
    
    Parameters:
    -----------
    X : array-like
        Input data
    y : array-like
        Labels for coloring
    components_list : list
        List of n_components to try
    n_runs : int
        Number of times to run each PCA for timing
    """
    # Create subplot grid with 5 plots per row
    n_plots = len(components_list)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 4*n_rows))
    gs = plt.GridSpec(n_rows, n_cols+1, width_ratios=[1]*n_cols + [0.05])
    
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Record running times
    running_times = []
    running_times_std = []  # Standard deviation of running times
    #Record variance explained
    var_explaineds = []
    
    # Create axes for plots
    axes = []
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        axes.append(plt.subplot(gs[row, col]))
    
    for idx, (ax, n_comp) in enumerate(zip(axes, components_list)):
        # Run PCA multiple times
        times = []
        for _ in range(n_runs):
            time_taken, pca, Z = run_pca_once(X, n_comp)
            times.append(time_taken)
        
        # Calculate mean and std of running times
        mean_time = np.mean(times)
        std_time = np.std(times)
        running_times.append(mean_time)
        running_times_std.append(std_time)
        
        # Use last PCA result for visualization
        if n_comp > 2:
            Z_2d = tsne.fit_transform(Z)
        else:
            Z_2d = Z
            
        # Plot
        scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='tab10', s=10)
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        var_explaineds.append(explained_var)
        ax.set_title(f'Standard-PCA (n={n_comp})\nVar: {explained_var:.1f}%\nTime: {mean_time:.3f}±{std_time:.3f}s')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    # Add colorbar in a separate axis
    cbar_ax = plt.subplot(gs[:, -1])
    plt.colorbar(scatter, cax=cbar_ax)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed results:")
    print(f"{'n_components':<12} {'Time (s)':<15} {'Std Dev':<10} {'Variance Explained':<20}")
    print("-" * 57)
    for n_comp, mean_time, std_time, var_explained in zip(components_list, running_times, running_times_std, var_explaineds):

        print(f"{n_comp:<12} {mean_time:<15.3f} {std_time:<10.3f} {var_explained:<20.2f}%")
    
    # Plot running times with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(components_list, running_times, yerr=running_times_std, 
                fmt='o-', linewidth=2, markersize=8, capsize=5)
    plt.xlabel('Number of Components')
    plt.ylabel('Running Time (seconds)')
    plt.title(f'PCA Running Time vs Number of Components\n(Average of {n_runs} runs with std dev)')
    plt.grid(True)
    plt.show()

# %%

components_to_try = [2,  5,  7, 10, 13, 15, 17, 20, 30, 40,]
compare_svd_pca_components(X, y, components_to_try, n_runs=3)

# %% [markdown]
# ### ML PPCA

# %%
def run_mlppca_once(X, n_comp):
    """
    Run ML-PPCA once and return running time and results
    
    Parameters:
    -----------
    X : array-like
        Input data
    n_comp : int
        Number of components
        
    Returns:
    --------
    time_taken : float
        Running time in seconds
    ml_ppca : MLPPCA object
        Fitted PPCA model
    Z : array-like
        Transformed data
    """
    start_time = time.time()
    ml_ppca = MLPPCA(n_components=n_comp)
    Z = ml_ppca.fit_transform(X)
    end_time = time.time()
    return end_time - start_time, ml_ppca, Z

def compare_mlpca_components(X, y, components_list=[2, 5, 10, 20], n_runs=3):
    """
    Compare ML-PPCA results with different numbers of components
    
    Parameters:
    -----------
    X : array-like
        Input data
    y : array-like
        Labels for coloring
    components_list : list
        List of n_components to try
    n_runs : int
        Number of times to run each PPCA for timing
    """
    # Create subplot grid with 5 plots per row
    n_plots = len(components_list)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 4*n_rows))
    gs = plt.GridSpec(n_rows, n_cols+1, width_ratios=[1]*n_cols + [0.05])
    
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Record running times
    running_times = []
    running_times_std = []  # Standard deviation of running times
    
    # Create axes for plots
    axes = []
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        axes.append(plt.subplot(gs[row, col]))
    
    for idx, (ax, n_comp) in enumerate(zip(axes, components_list)):
        # Run ML-PPCA multiple times
        times = []
        for _ in range(n_runs):
            time_taken, ml_ppca, Z = run_mlppca_once(X, n_comp)
            times.append(time_taken)
        
        # Calculate mean and std of running times
        mean_time = np.mean(times)
        std_time = np.std(times)
        running_times.append(mean_time)
        running_times_std.append(std_time)
        
        # Use last PPCA result for visualization
        if n_comp > 2:
            Z_2d = tsne.fit_transform(Z)
        else:
            Z_2d = Z
            
        # Plot
        scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='tab10', s=10)
        ax.set_title(f'ML-PPCA (n={n_comp})\nTime: {mean_time:.3f}±{std_time:.3f}s')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    # Add colorbar in a separate axis
    cbar_ax = plt.subplot(gs[:, -1])
    plt.colorbar(scatter, cax=cbar_ax)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed results:")
    print(f"{'n_components':<12} {'Time (s)':<15} {'Std Dev':<10}")
    print("-" * 37)
    for n_comp, mean_time, std_time in zip(components_list, running_times, running_times_std):
        print(f"{n_comp:<12} {mean_time:<15.3f} {std_time:<10.3f}")
    
    # Plot running times with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(components_list, running_times, yerr=running_times_std, 
                fmt='o-', linewidth=2, markersize=8, capsize=5)
    plt.xlabel('Number of Components')
    plt.ylabel('Running Time (seconds)')
    plt.title(f'ML-PPCA Running Time vs Number of Components\n(Average of {n_runs} runs with std dev)')
    plt.grid(True)
    plt.show()

# 使用示例
components_to_try = [2,  5,  7, 10, 13, 15, 17, 20, 30, 40,]
compare_mlpca_components(X, y, components_to_try, n_runs=3)

# %% [markdown]
# ### EM PPCA

# %%
def run_ppca_em_once(X, n_comp, n_iter=30, seed=None):
    """
    Run PPCA EM algorithm once and return running time and results
    
    Parameters:
    -----------
    X : array-like
        Input data
    n_comp : int
        Number of components
    n_iter : int
        Number of EM iterations
    seed : int
        Random seed for initialization
        
    Returns:
    --------
    time_taken : float
        Running time in seconds
    ppca : ProbabilisticPCA object
        Fitted PPCA model
    Z : array-like
        Transformed data
    Q_history : list
        History of log-likelihood values
    """
    start_time = time.time()
    
    # Initialize PPCA
    ppca = ProbabilisticPCA(X, M=n_comp, seed=seed)
    
    # Run EM iterations
    Q_history = [ppca.data_log_likelihood()]
    for _ in range(n_iter):
        Q = ppca.EM_step()
        Q_history.append(Q)
        # Stop if Q converges
        if len(Q_history) > 1 and abs(Q_history[-1] - Q_history[-2]) < 1e-3:
            break
    
    # Get final projection
    Z = ppca.project().T
    
    end_time = time.time()
    return end_time - start_time, ppca, Z, Q_history

def compare_ppca_em_components(X, y, components_list=[2, 5, 10, 20], n_runs=3, n_iter=30):
    """
    Compare PPCA EM results with different numbers of components
    
    Parameters:
    -----------
    X : array-like
        Input data
    y : array-like
        Labels for coloring
    components_list : list
        List of n_components to try
    n_runs : int
        Number of times to run each PPCA for timing
    n_iter : int
        Maximum number of EM iterations
    """
    # Create subplot grid with 5 plots per row
    n_plots = len(components_list)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 4*n_rows))
    gs = plt.GridSpec(n_rows, n_cols+1, width_ratios=[1]*n_cols + [0.05])
    
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Record running times and convergence iterations
    running_times = []
    running_times_std = []
    convergence_iters = []
    convergence_iters_std = []
    
    # Create axes for plots
    axes = []
    for i in range(n_plots):
        row = i // n_cols
        col = i % n_cols
        axes.append(plt.subplot(gs[row, col]))
    
    for idx, (ax, n_comp) in enumerate(zip(axes, components_list)):
        # Run PPCA EM multiple times
        times = []
        iters = []
        for run in range(n_runs):
            time_taken, ppca, Z, Q_history = run_ppca_em_once(X, n_comp, n_iter, seed=42+run)
            times.append(time_taken)
            iters.append(len(Q_history)-1)  # -1 because Q_history includes initial value
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_iters = np.mean(iters)
        std_iters = np.std(iters)
        
        running_times.append(mean_time)
        running_times_std.append(std_time)
        convergence_iters.append(mean_iters)
        convergence_iters_std.append(std_iters)
        
        # Use last PPCA result for visualization
        if n_comp > 2:
            Z_2d = tsne.fit_transform(Z)
        else:
            Z_2d = Z
            
        # Plot
        scatter = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=y, cmap='tab10', s=10)
        ax.set_title(f'PPCA EM (n={n_comp})\nTime: {mean_time:.3f}±{std_time:.3f}s\nIters: {mean_iters:.1f}±{std_iters:.1f}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    # Add colorbar in a separate axis
    cbar_ax = plt.subplot(gs[:, -1])
    plt.colorbar(scatter, cax=cbar_ax)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nDetailed results:")
    print(f"{'n_components':<12} {'Time (s)':<15} {'Std Dev':<10} {'Iterations':<15} {'Iter Std Dev':<10}")
    print("-" * 62)
    for n_comp, mean_time, std_time, mean_iter, std_iter in zip(
        components_list, running_times, running_times_std, convergence_iters, convergence_iters_std):
        print(f"{n_comp:<12} {mean_time:<15.3f} {std_time:<10.3f} {mean_iter:<15.1f} {std_iter:<10.1f}")
    
    # Plot running times with error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Time plot
    ax1.errorbar(components_list, running_times, yerr=running_times_std, 
                fmt='o-', linewidth=2, markersize=8, capsize=5)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Running Time (seconds)')
    ax1.set_title(f'PPCA EM Running Time\n(Average of {n_runs} runs with std dev)')
    ax1.grid(True)
    
    # Iterations plot
    ax2.errorbar(components_list, convergence_iters, yerr=convergence_iters_std,
                fmt='o-', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Number of Iterations')
    ax2.set_title(f'PPCA EM Convergence Iterations\n(Average of {n_runs} runs with std dev)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 使用示例
components_to_try = components_to_try = [2,  5,  7, 10, 13, 15, 17, 20, 30, 40,]
compare_ppca_em_components(X, y, components_to_try, n_runs=3, n_iter=50)

# %%



