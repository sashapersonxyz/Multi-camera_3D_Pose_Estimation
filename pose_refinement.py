import os
import numpy as np
import torch
import utils
import time
import argparse
import yaml
from pathlib import Path
import pickle as pk





def linear_interpolation(points, k=5, k_std=2, median_std=2, use_rolling_average=False, filter_distance_from_median = True):
    """
    Smooth points by filtering outliers and interpolating.
    
    points: array of shape [time, n_points, dim] or [time, n_points]
    k: window size
    k_std: number of std deviations for mean-based filtering
    median_std: number of std deviations for median-based filtering
    use_rolling_average: if True, uses mean of valid points; otherwise interpolates
    """
    points = np.array(points)
    time_index, point_index = points.shape[:2]
    dim = points.shape[2] if len(points.shape) == 3 else 1  # Treat as 1D if [time, n_points]
    approximated_points = np.zeros_like(points)

    for p in range(point_index):
        for d in range(dim):
            for t in range(time_index):
                # Define a symmetric window of size ~k centered at t
                window_start = max(0, t - k // 2)
                window_end = min(time_index, t + k // 2 + 1)
                
                # Extract points in window
                window_points = (
                    points[window_start:window_end, p]
                    if dim == 1 else points[window_start:window_end, p, d]
                )
                
                # Stats for filtering
                mean = np.mean(window_points)
                std = np.std(window_points)
                median = np.median(window_points)
                mad = np.median(np.abs(window_points - median))  # median abs dev

                if filter_distance_from_median:
                    # Keep points within both mean±k_std*std AND median±median_std*mad
                    valid_indices = (
                        (np.abs(window_points - mean) <= k_std * std) &
                        (np.abs(window_points - median) <= median_std * mad)
                    )
                else:
                    valid_indices = (np.abs(window_points - mean) <= k_std * std)
                valid_points = window_points[valid_indices]
                
                # Handle edge cases where valid points are insufficient
                new_point = 0
                if len(valid_points) < 2:
                    new_point = points[t, p] if dim == 1 else points[t, p, d]
                    continue
                
                if use_rolling_average:
                    # Use the average of valid points in the window
                    new_point = np.mean(valid_points)
                else:
                    # Perform linear interpolation using the valid points and their corresponding times
                    valid_times = np.arange(window_start, window_end)[valid_indices]
                    
                    # Ensure we fit the line even with few points
                    if len(valid_points) > 1:
                        coef = np.polyfit(valid_times, valid_points, 1)  # Linear fit
                        new_point = np.polyval(coef, t)  # Predict at time t
                    else:
                        # If insufficient points for polyfit, fallback to using the mean
                        new_point = np.mean(valid_points)
    
                if dim == 1:
                    approximated_points[t, p] = new_point
                else:
                    approximated_points[t,p,d] = new_point
    return approximated_points








#project 3d trajectory to camera coordinates
def project_points_torch(points, K, R, T, dist_coeffs, indicies = None, torch_dtype = torch.float32, ignore_distortions = False):
    '''points is of shape (Time, N, 3)'''
    
    # Convert inputs if not a tensor or if it's not the right dtype
    if not isinstance(R, torch.Tensor):
        R = torch.tensor(R, dtype=torch_dtype)
    if R.dtype != torch_dtype: R = R.to(dtype=torch_dtype)
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch_dtype)
    if K.dtype != torch_dtype: K = K.to(dtype=torch_dtype)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch_dtype)
    if T.dtype != torch_dtype: T = T.to(dtype=torch_dtype)
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch_dtype)
    if points.dtype != torch_dtype: points = points.to(dtype=torch_dtype)    
    # Convert dist_coeffs if it's not a tensor or if it's not the right dtype
    if not isinstance(dist_coeffs, torch.Tensor) or (isinstance(dist_coeffs, torch.Tensor) and dist_coeffs.dtype != torch_dtype):
        dist_coeffs = torch.tensor(dist_coeffs, dtype=torch_dtype)   
    
    R = utils.rotation_conversion(R, to_vector=False)
        
    # Ensure the input shapes are compatible
    assert points.shape[2] == 3, "points must have shape (Time, N, 3)"
    assert K.shape == (3, 3), "K must have shape (3, 3)"
    assert R.shape == (3, 3), "R must have shape (3, 3)"
    assert T.shape == (3, 1) or T.shape == (3,), "T must have shape (3,) or (3, 1)"
    assert dist_coeffs.shape == (1, 5), "dist_coeffs must have shape (1, 5)"

    # Reshape the trajectory from (Time, N, 3) to (Time*N, 3)
    time, n_points, _ = points.shape
    if indicies is None:
        indicies = list(range(time))
    time = len(indicies)
    trajectory_reshaped = points[indicies].reshape(time * n_points, 3)

    # Flatten T if necessary
    T = T.squeeze()  # Shape will be (3,) if it was (3, 1)
    
    # Homogeneous coordinates
    ones = torch.ones(trajectory_reshaped.shape[0], 1, device=trajectory_reshaped.device, dtype = torch_dtype)
    trajectory_hom = torch.cat([trajectory_reshaped, ones], dim=-1)  # Shape (Time*N, 4)

    # Extrinsic matrix [R | T]
    extrinsic = torch.cat([R, T.unsqueeze(-1)], dim=-1)  # Shape (3, 4)

    # Transform 3D points into camera coordinates
    point_camera = trajectory_hom @ extrinsic.T  # Shape (Time*N, 3)

    # Normalize to get the image plane coordinates (camera coordinates)
    x_normalized = point_camera[:, 0] / point_camera[:, 2]
    y_normalized = point_camera[:, 1] / point_camera[:, 2]
    
    if not ignore_distortions:
        # Apply radial and tangential distortion
        r2 = x_normalized**2 + y_normalized**2  
        
        # Extract distortion coefficients
        k1, k2, p1, p2, k3 = dist_coeffs.squeeze()
        
        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x_distorted = x_normalized * radial
        y_distorted = y_normalized * radial
    
        # Tangential distortion
        x_distorted += 2 * p1 * x_normalized * y_normalized + p2 * (r2 + 2 * x_normalized**2)
        y_distorted += p1 * (r2 + 2 * y_normalized**2) + 2 * p2 * x_normalized * y_normalized
        
        # Apply intrinsic matrix K to map distorted normalized coordinates to pixel coordinates
        point_2d = torch.stack([x_distorted, y_distorted], dim=-1)  # (Time*N, 2)

    else:
        point_2d = torch.stack([x_normalized, y_normalized], dim=-1)  # (Time*N, 2)
            
    # Convert from normalized camera coordinates to pixel coordinates
    point_2d_hom = torch.cat([point_2d, torch.ones(point_2d.shape[0], 1, dtype=torch_dtype)], dim=-1)  # (Time*N, 3)          
    point_2d_pixel = point_2d_hom @ K.T  # (Time*N, 3)

    # Normalize by the last coordinate to get (x, y) in pixel coordinates
    point_2d_pixel = point_2d_pixel[:, :2] / point_2d_pixel[:, 2].unsqueeze(-1)

    # Reshape back to (Time, N, 2)
    point_2d_pixel = point_2d_pixel.reshape(time, n_points, 2)
    
    return point_2d_pixel  # (Time, N, 2)


def gaussian_likelihood(x, mean, cov_mat, eps=1e-6, torch_dtype=torch.float32):
    """
    Computes the log-likelihood of `x` given the 2D Gaussian with the provided mean and covariance matrix.
    
    Args:
    - x: Input data point (Tensor) with shape [d0, ..., dk, 2].
    - mean: Mean of the Gaussian (Tensor) with shape [d0, ..., dk, 2].
    - cov_mat: Covariance matrix of the Gaussian (Tensor) with shape [d0, ..., dk, 2, 2].
    - eps: Small value to ensure numerical stability (float, default: 1e-6).
    - torch_dtype: Data type for the computations (default: torch.float32).
    
    Returns:
    - log_likelihood: Log-likelihood of `x` under the Gaussian distribution with shape [d0, ..., dk].
    """
    # Ensure covariance matrix is positive definite
    cov_mat = cov_mat + eps * torch.eye(cov_mat.size(-1), device=cov_mat.device).expand_as(cov_mat)

    # Compute the inverse of the covariance matrix
    cov_inv = torch.linalg.inv(cov_mat).to(torch_dtype)
    
    # Compute the difference between x and the mean, with shape [d0, ..., dk, 2]
    diff = x - mean
    
    # Compute the quadratic term: -0.5 * (diff^T * cov_inv * diff)
    # Use torch.einsum for batch processing: '...i,...ij,...j->...' performs the matrix multiplication
    quadratic_term = -0.5 * torch.einsum('...i,...ij,...j->...', diff, cov_inv, diff)
    
    # Compute the determinant of the covariance matrix
    cov_det = torch.det(cov_mat)
    
    # Compute the normalization term: 0.5 * log((2 * pi) ^ 2 * det(cov_mat))
    normalization_term = 0.5 * torch.log((2 * torch.pi) ** 2 * cov_det + eps)
    
    # Combine quadratic term and normalization term to compute the log-likelihood
    log_likelihood = quadratic_term - normalization_term
    
    return log_likelihood


def nan_mean(X):
    '''computes torch.nanmean while preserving gradients
    '''
    stacked_tensor = torch.stack(X)
    
    # Check for NaN or infinity using the OR operator
    index_mask = ~(torch.isnan(stacked_tensor) | torch.isinf(stacked_tensor))

    return torch.sum(stacked_tensor[index_mask])/len(stacked_tensor[index_mask])



class ExtrinsicParameterRefinement:
    def __init__(self, gaussians, R_initial=None, T_initial=None, decomposed_cam_params = None, N_sample_points = 100, GT_camera_indicies = [0,1], estimation_camera_index = 2, torch_dtype = torch.float32):
        assert len(GT_camera_indicies) == 2
        assert min([idx in decomposed_cam_params.keys() for idx in GT_camera_indicies])
        # Initialize R and T as torch tensors with gradients enabled
        if R_initial is None and T_initial is None:
            if estimation_camera_index in decomposed_cam_params:
                self.R = torch.tensor(decomposed_cam_params[estimation_camera_index][1], dtype=torch.float32)
                self.T = torch.tensor(decomposed_cam_params[estimation_camera_index][2], dtype=torch.float32)
            else:
                self.R = torch.eye(3) if R_initial is None else torch.tensor(R_initial, dtype=torch.float32)
                self.T = torch.zeros(3, 1) if T_initial is None else torch.tensor(T_initial, dtype=torch.float32)
        else:
            self.R = torch.eye(3)
            self.T = torch.zeros(3, 1)
            
        self.R = self.R.requires_grad_(True)
        self.T = self.T.requires_grad_(True)
        
        self.gaussians = torch.tensor(gaussians, dtype=torch_dtype)
        self.decomposed_cam_params = decomposed_cam_params
        if self.decomposed_cam_params is not None:
            self.decomposed_cam_params = dict(zip(decomposed_cam_params.keys(), [[torch.tensor(CP, dtype=torch_dtype) for CP in decomposed_cam_params[key]] for key in decomposed_cam_params.keys()]))
        self.Time = gaussians.shape[0]
        self.n_cams = gaussians.shape[1]
        
        assert self.n_cams == 3
        
        self.n_joints = gaussians.shape[2]
        self.N_sample_points = N_sample_points
        self.GT_camera_indicies = GT_camera_indicies
        self.estimation_camera_index = estimation_camera_index
        self.torch_dtype = torch_dtype
        
    def sample_gaussians(self, N = None):
        if N==None:
            N = self.N_sample_points
        means = self.gaussians[:,self.GT_camera_indicies,:, :2]  # Shape: [Time, 2, n_joints, 2]
        cov_matrices = self.gaussians[:,self.GT_camera_indicies,:, 2:].reshape(self.Time, 2, self.n_joints, 2, 2)  # Shape: [Time, 2, n_joints, 2, 2]
        
        # Initialize tensor to store samples
        samples = np.empty((self.Time, 2, self.n_joints, N, 2))
        
        # Sampling loop
        for t in range(self.Time):
            for cam in range(2):
                for point in range(self.n_joints):
                    mean = means[t, cam, point]  # Shape: [2]
                    cov = cov_matrices[t, cam, point]  # Shape: [2, 2]
                    
                    # Sample N points from the Gaussian distribution
                    samples[t, cam, point] = np.random.multivariate_normal(mean.numpy(), cov.numpy(), N)
                    
        # Now samples is of shape [Time, 2, n_points, N, 2], so put n_cams second to last for utils.triangulate_points
        samples = np.transpose(samples, (0,2,3,1,4))
        self.samples = samples
        return samples

    def construct_loss(self):
        #collect estimation heatmaps
        means = self.gaussians[:,2,:, :2]  # Shape: [Time, 2, n_points, 2]
        cov_matrices = self.gaussians[:,2,:, 2:].reshape(self.Time, 1, self.n_joints, 2, 2)  # Shape: [Time, 2, n_points, 2, 2]
        
        cmtx1, R1, T1, dist1 = self.decomposed_cam_params[self.GT_camera_indicies[0]]
        cmtx2, R2, T2, dist2 = self.decomposed_cam_params[self.GT_camera_indicies[1]]
        

        self.samples_3d = utils.triangulate_points(self.samples, cmtx1, dist1, R1, T1, cmtx2, dist2, R2, T2)
        self.samples_3d = torch.from_numpy(self.samples_3d).to(self.torch_dtype)
        def loss(R, T):
            likelihoods = []
            for n in range(self.N_sample_points):
                sample_3d = self.samples_3d[:,:,n,:].squeeze()
                points_2d = project_points_torch(sample_3d, self.decomposed_cam_params[self.estimation_camera_index][0], R, T, self.decomposed_cam_params[self.estimation_camera_index][-1], torch_dtype=self.torch_dtype)        
                likelihoods.append(gaussian_likelihood(points_2d, means, cov_matrices, eps=1e-6, torch_dtype = torch.float32))

            mean_likelihood = nan_mean(likelihoods)
            return mean_likelihood
        self.loss_function = loss
        return loss

    def optimize(self, learning_rate=0.001, print_frequency = 10, max_iter = np.inf, patience = 10):
        self.sample_gaussians()
        self.construct_loss()
        # Create the Adam optimizer
        self.optimizer = torch.optim.Adam([self.R, self.T], lr=learning_rate)
        
        
        
        best_cost = float('inf')
        iteration=0
        no_improvement_count=0
        while no_improvement_count < patience and iteration <= max_iter:  # Set a very large number of max iterations

            self.optimizer.zero_grad()

            cost = self.loss_function(self.R, self.T)


            # Backpropagate to compute gradients
            cost.backward()
            
            # Update R and T
            self.optimizer.step()

            # Optional: Re-orthogonalize R to maintain it as a valid rotation matrix
            with torch.no_grad():
                U, _, Vt = torch.linalg.svd(self.R)
                self.R.copy_(U @ Vt)
                
                
            # Check for improvement
            current_cost = cost.clone().detach()
            if current_cost < best_cost:
                best_cost = current_cost
                self.best_params = [self.R.clone().detach(), self.T.clone().detach()]
                no_improvement_count = 0  # Reset no improvement counter
            else:
                no_improvement_count += 1
            
            # Early stopping if no improvement for `patience` iterations
            if no_improvement_count >= patience:
                print(f"Early stopping at iteration {iteration}. Best cost = {best_cost:.2e}")
                break
            # Optionally, print the current cost for monitoring
            if iteration % print_frequency == 0:
                print(f"Iteration {iteration}: Cost = {current_cost:.2e}")
            iteration+=1

        return self.best_params



import torch
import matplotlib.pyplot as plt
import random
import torch.nn as nn
class Trajectory_Optimization:
    def __init__(self, gaussians, initial_trajectory, decomposed_cam_params = None, body_lengths = None, torch_dtype = torch.float32, camera_IDs = None):

        
        
        """
        Optimizes the trajectory using Adam with early stopping.
        
        Args:
        - gaussians: Tensor of shape (Time, C_camera_index, n_joints, 6) [(Time, C_camera_index, n_joints, 4)] representing means and cov [std] of the camera projections.
        - initial_trajectory: Initial guess for the trajectory (Time, n_joints, 3) [(Time, n_joints, 2) for 2D].
        - decomposed_cam_params: Camera parameters (K, R, T, dist_coeffs)
        """

        self.camera_IDs = camera_IDs if camera_IDs is not None else list(decomposed_cam_params.keys())
        self.camera_indices = [list(decomposed_cam_params.keys()).index(ID) for ID in self.camera_IDs] 
        
        
        self.gaussians = torch.tensor(gaussians, dtype=torch_dtype)
        self.initial_trajectory = torch.tensor(initial_trajectory, dtype=torch_dtype)
        self.decomposed_cam_params = decomposed_cam_params
        if self.decomposed_cam_params is not None:
            self.decomposed_cam_params = dict(zip(decomposed_cam_params.keys(), [[torch.tensor(CP, dtype=torch_dtype) for CP in decomposed_cam_params[key]] for key in decomposed_cam_params.keys()]))
        

        self.torch_dtype = torch_dtype
        self.n_dims = initial_trajectory.shape[2]
        self.Time = gaussians.shape[0]
        self.n_cams = len(self.camera_indices)
        self.n_joints = gaussians.shape[2]
        self.body_lengths = body_lengths
    
    def create_body_length_vect(self):
        # Extract values from the dictionary and convert them to a list
        BPL = list(self.body_lengths.values())
        
        # Convert the list to a tensor
        BPL = torch.tensor(BPL, dtype=self.torch_dtype).unsqueeze(1)
        
        # Repeat each value N times before switching to the next
        repeated_tensor = BPL.repeat_interleave(self.batch_size, dim=0)
        
        # Reshape to [Time*|BPL|, 1]
        final_tensor = repeated_tensor.view(self.batch_size * len(BPL), 1).squeeze()
        
        return final_tensor
    
         
        
        
    def create_batch_indices(self):
        # List to store the batches
        batches = []

#       step_size =  self.batch_size
        step_size = self.batch_size // 2  # Overlap by half the batch size
        
        for start in range(0, self.Time - self.batch_size + 1, step_size):
            batch = list(range(start, start + self.batch_size))
            batches.append(batch)
        return batches


    
    
    def compute_smoothness_cost(self, indicies):
        """Computes the smoothness term of the cost function."""
        self.smoothness_costs = []
#        smoothness_cost = 0.0
        for idx in range(2, len(indicies)):
            t = indicies[idx]
            diff = (self.trajectory[t] - self.trajectory[t - 1]) - (self.trajectory[t - 1] - self.trajectory[t - 2])
            self.smoothness_costs.append(torch.norm(diff) ** 2)
 #           smoothness_cost += torch.norm(diff) ** 2
        return nan_mean(self.smoothness_costs)
    
    
    def compute_body_length_cost(self, indicies):
        BPL = utils.get_body_part_lengths(self.trajectory[indicies])
        
        BPL_stacked = torch.hstack([BPL[bl] for bl in self.body_lengths.keys()]).squeeze()
        
        # minimum cost mu: mu = (a^T b) / (||b||^2)
        mu_val = (torch.dot(self.body_length_vect, BPL_stacked)) / torch.dot(BPL_stacked, BPL_stacked)
        
        # Calculate a - lambda * b
        diff = self.body_length_vect - mu_val * BPL_stacked
        
        # Return the squared norm of the difference (divided by |a|^2 for normalization)
        return torch.norm(diff) ** 2 / torch.norm(self.body_length_vect) ** 2
        
    
    
    
    
    # def likelihood_term(trajectory, gaussians):
    #     """Computes the likelihood term based on the Gaussian parameters (mean, std)."""
    #     likelihood_cost = 0.0
    #     T = trajectory.shape[0]
        
    #     for t in range(T):
    #         mean = gaussians[t, 0, :]
    #         std = gaussians[t, 1, :]
    #         log_likelihood = gaussian_likelihood(trajectory[t], mean, std)
    #         likelihood_cost -= log_likelihood
        
    #     return likelihood_cost
    
    
    
    def compute_total_cost(self, lambda_smooth, lambda_body_length, indicies = None):
        if indicies == None:
            indicies = range(self.Time)
        
        """Computes the total cost for a given trajectory."""
        costs = []
        self.smoothness_cost = lambda_smooth * self.compute_smoothness_cost(indicies)
        costs.append(self.smoothness_cost)
        
        
        self.body_length_cost = lambda_body_length * self.compute_body_length_cost(indicies)
        
        costs.append(self.body_length_cost)
        self.likelihood_costs = []
        for camera_index, camera_ID in zip(self.camera_indices, self.camera_IDs):
            cam_params = self.decomposed_cam_params[camera_ID]
            trajectory_projection = project_points_torch(self.trajectory, *cam_params, indicies = indicies, torch_dtype=self.torch_dtype) if self.n_dims == 3 else self.trajectory[indicies]
            for idx, T in enumerate(indicies):
                for joint in range(self.n_joints):
                        gauss_params = self.gaussians[T, camera_index, joint, :]
                        mean = gauss_params[:2]
                        cov_mat = gauss_params[2:].reshape([2,2]) if len(gauss_params[2:])==4 else torch.diag(gauss_params[2:])
    
                        
                        likelihood_cost = gaussian_likelihood(trajectory_projection[idx, joint, :], mean, cov_mat)
                        self.likelihood_costs.append(likelihood_cost)
        self.total_likelihood_cost = nan_mean(self.likelihood_costs)
        costs.append(self.total_likelihood_cost)
        
        self.total_cost = torch.sum(torch.stack(costs))
    
    def sgd_optimize(self, lr=0.001, betas=(0.9, 0.999), lambda_smooth=1.0, lambda_body_length=1.0, patience=10, tolerance=1e-5, max_iter = np.inf, print_frequency = 100, batch_size = None):
        
        if batch_size is None:
            batch_size = self.Time
        self.batch_size = batch_size
        
        #just make batches consistent for now
        self.Time = int(np.floor(self.Time/batch_size)*batch_size)
        self.initial_trajectory = self.initial_trajectory[:self.Time]
        
        # Initialize the trajectory with the provided initial trajectory
        self.trajectory = self.initial_trajectory.clone().detach().requires_grad_(True)

        batch_indicies = self.create_batch_indices()
        
        self.body_length_vect = self.create_body_length_vect()
        
        optimizer = torch.optim.Adam([self.trajectory], lr=lr, betas = betas)
        
        
        best_cost = [float('inf'), float('inf'), float('inf')]
        self.best_trajectory = None
        no_improvement_count = 0
        
        iteration=0
        while no_improvement_count < patience and iteration <= max_iter:  # Set a very large number of max iterations
            optimizer.zero_grad()
            all_costs = []
            for idx, batch in enumerate(batch_indicies):
                # Compute the cost
                self.compute_total_cost(lambda_smooth, lambda_body_length, indicies=batch)
                
                # Perform backward pass
                self.total_cost.backward()
        
                # Clip gradients to avoid instability
                torch.nn.utils.clip_grad_norm_([self.trajectory], max_norm=1.0)
                
                # Update trajectory using the optimizer
                optimizer.step()
                
                all_costs.append([self.total_cost.item(), self.smoothness_cost.item(), self.total_likelihood_cost.item(), self.body_length_cost.item()])
                
            # Check for improvement
            current_cost, current_smoothness_cost, current_likelihood_cost, current_body_length_cost = np.sum(all_costs, 0)
            if current_cost < best_cost[0] - tolerance:
                best_cost = [current_cost, current_smoothness_cost, current_likelihood_cost, current_body_length_cost]
                self.best_trajectory = self.trajectory.clone().detach()
                no_improvement_count = 0  # Reset no improvement counter
            else:
                no_improvement_count += 1
            
            # Early stopping if no improvement for `patience` iterations
            if no_improvement_count >= patience:
                print(f"Early stopping at iteration {iteration}. Best cost, smoothness, likelihood = {best_cost:.2e}")
                break
            # Optionally, print the current cost for monitoring
            if iteration % print_frequency == 0:
                print(f"Iteration {iteration}: Total Cost = {current_cost:.2e}, Smoothness Cost = {self.smoothness_cost:.2e}, Likelihood Cost = {self.total_likelihood_cost:.6e}, Body Length Cost = {self.body_length_cost:.2e}")
            iteration+=1

    


class Optimized_3d_Pose_Estimation:
    
     
    
    def __init__(self, gaussians, initial_trajectory, decomposed_cam_params_initial = None, body_lengths = None, camera_IDs = None, R_initial=None, T_initial=None, N_sample_points = 100, torch_dtype = torch.float32):
                
        """
        Object to optimize trajectory and/or camera parameters
        
        Args:
        - gaussians: Tensor of shape (Time, C_camera_index, n_joints, 6) [(Time, C_camera_index, n_joints, 4)] representing means and cov [std] of the camera projections.
        - initial_trajectory: Initial guess for the trajectory (Time, n_joints, 3) [(Time, n_joints, 2) for 2D].
        - decomposed_cam_params: Camera parameters (K, R, T, dist_coeffs)
        """


        # Add the new layer for transforming each (3, 6) sample to (3,)
        self.simple_nn = nn.Sequential(
            nn.Flatten(start_dim=-2),    # Flatten (3, 6) to (18)
            nn.Linear(18, 256),           # First linear layer
            nn.ReLU(),                   # Activation
            nn.Linear(256, 128),           # First linear layer
            nn.ReLU(),                   # Activation
            nn.Linear(128, 64),           # First linear layer
            nn.ReLU(),                   # Activation
            nn.Linear(64, 32),             # Output layer to get (3,)
            nn.ReLU(),                   # Activation
            nn.Linear(32, 16),             # Output layer to get (3,)
            nn.ReLU(),                   # Activation
            nn.Linear(16, 3)             # Output layer to get (3,)
        )




        for camera_index in decomposed_cam_params_initial:
            if decomposed_cam_params_initial[camera_index][1] is None:
                decomposed_cam_params_initial[camera_index][1] = torch.eye(3)
            if decomposed_cam_params_initial[camera_index][2] is None:
                decomposed_cam_params_initial[camera_index][2] = torch.zeros(3, 1)


                
        self.gaussians = torch.tensor(gaussians, dtype=torch_dtype)
        self.decomposed_cam_params_initial = decomposed_cam_params_initial
        if self.decomposed_cam_params_initial is not None:
            self.decomposed_cam_params_initial = dict(zip(decomposed_cam_params_initial.keys(), [[torch.tensor(CP, dtype=torch_dtype) for CP in decomposed_cam_params_initial[key]] for key in decomposed_cam_params_initial.keys()]))
        
        self.decomposed_cam_params = {camera_ID:[cam_param.clone().detach() for cam_param in self.decomposed_cam_params_initial[camera_ID]] for camera_ID in self.decomposed_cam_params_initial}
        self.n_cams = gaussians.shape[1]
        
        self.N_sample_points = N_sample_points
        self.initial_trajectory = torch.tensor(initial_trajectory, dtype=torch_dtype)
        self.torch_dtype = torch_dtype
        self.n_dims = initial_trajectory.shape[2]
        self.n_joints = gaussians.shape[2]
        self.body_lengths = body_lengths
        self.camera_IDs = camera_IDs if camera_IDs is not None else list(decomposed_cam_params_initial.keys())
        self.camera_indices = [list(self.decomposed_cam_params.keys()).index(ID) for ID in self.camera_IDs]
        
        
        
        #precompute cov^-1
        if not hasattr(self, 'cov_invs'):
            self.cov_invs = dict()
            for camera_index, camera_ID in zip(self.camera_indices, self.camera_IDs):
                # if camera_ID in self.extrinsic_optimization_IDs:
                #     self.cov_invs[camera_ID] = None
                # else:
                eps=1e-6
                # self.cov_invs[camera_ID] = []

                # cov_invs_per_time = []                
                # for T in range(self.Time):
                #     cov_invs_per_joint = []    
                #     for joint in range(self.n_joints):
                #         gauss_params = self.gaussians_subset[T, camera_index, joint, :]
                #         cov_mat = gauss_params[2:].reshape([2,2]) if len(gauss_params[2:])==4 else torch.diag(gauss_params[2:])
            
                #         # Ensure covariance matrix is positive definite
                #         cov_mat = cov_mat + eps * torch.eye(cov_mat.size(-1), device=cov_mat.device).expand_as(cov_mat)
                    
                #         # Compute the inverse of the covariance matrix
                #         cov_inv = torch.linalg.inv(cov_mat).to(self.torch_dtype)
                #         cov_invs_per_joint.append(cov_inv)
                #     cov_invs_per_time.append(cov_invs_per_joint)
                # self.cov_invs.append(cov_invs_per_time)
                
                cov_mat = self.gaussians[:,0,:,2:].reshape(self.gaussians.shape[0],self.n_joints,2,2)
                cov_mat = cov_mat + eps * torch.eye(cov_mat.size(-1), device=cov_mat.device).expand_as(cov_mat)
            
                # Compute the inverse of the covariance matrix
                cov_inv = torch.linalg.inv(cov_mat).to(self.torch_dtype)
                self.cov_invs[camera_ID]  = cov_inv
        
        
        
    def forward_NN(self, x):
        # Expected input shape: [n_samples_time, 3, n_samples_points, 6]
        
        # Reshape to treat each (3, 6) sample independently
        n_samples_time, _, n_samples_points, _ = x.shape
        x = x.permute(0, 2, 1, 3)  # Reorder to [n_samples_time, n_samples_points, 3, 6]
        
        # Apply the transformation across each (3, 6) sample
        x = self.simple_nn(x)       # Now [n_samples_time, n_samples_points, 3]
        
        # Return the output with the desired shape
        return x
    def sample_gaussians(self, N = None):
        if N==None:
            N = self.N_sample_points
        means = self.gaussians_subset[:,self.GT_camera_IDs,:, :2]  # Shape: [Time, 2, n_joints, 2]
        cov_matrices = self.gaussians_subset[:,self.GT_camera_IDs,:, 2:].reshape(self.Time, 2, self.n_joints, 2, 2)  # Shape: [Time, 2, n_joints, 2, 2]
        
        # Initialize tensor to store samples
        samples = np.empty((self.Time, 2, self.n_joints, N, 2))
        
        # Sampling loop
        for t in range(self.Time):
            for cam in range(2):
                for point in range(self.n_joints):
                    mean = means[t, cam, point]  # Shape: [2]
                    cov = cov_matrices[t, cam, point]  # Shape: [2, 2]
                    
                    # Sample N points from the Gaussian distribution
                    samples[t, cam, point] = np.random.multivariate_normal(mean.numpy(), cov.numpy(), N)
                    
        # Now samples is of shape [Time, 2, n_points, N, 2], so put n_cams second to last for utils.triangulate_points
        samples = np.transpose(samples, (0,2,3,1,4))
        self.samples = samples
        return samples




    
    def gaussian_likelihood(self, x, mean, cov_mat, eps=1e-6, cov_inv = None):
        """
        Computes the log-likelihood of `x` given the 2D Gaussian with the provided mean and covariance matrix.
        
        Args:
        - x: Input data point (Tensor) with shape [d0, ..., dk, 2].
        - mean: Mean of the Gaussian (Tensor) with shape [d0, ..., dk, 2].
        - cov_mat: Covariance matrix of the Gaussian (Tensor) with shape [d0, ..., dk, 2, 2].
        - eps: Small value to ensure numerical stability (float, default: 1e-6).
        - torch_dtype: Data type for the computations (default: torch.float32).
        
        Returns:
        - log_likelihood: Log-likelihood of `x` under the Gaussian distribution with shape [d0, ..., dk].
        """
        
        #if we have an inverse precomputed, we can use it and ignore the normalization term
        if cov_inv is not None:
                
            # Compute the difference between x and the mean, with shape [d0, ..., dk, 2]
            diff = x - mean
            
            # Compute the quadratic term: -0.5 * (diff^T * cov_inv * diff)
            # Use torch.einsum for batch processing: '...i,...ij,...j->...' performs the matrix multiplication
            quadratic_term = -0.5 * torch.einsum('...i,...ij,...j->...', diff, cov_inv, diff)
            normalization_term = 0
        else:
            # Ensure covariance matrix is positive definite
            cov_mat = cov_mat + eps * torch.eye(cov_mat.size(-1), device=cov_mat.device).expand_as(cov_mat)
        
            # Compute the inverse of the covariance matrix
            cov_inv = torch.linalg.inv(cov_mat).to(self.torch_dtype)
            
            # Compute the difference between x and the mean, with shape [d0, ..., dk, 2]
            diff = x - mean
            
            # Compute the quadratic term: -0.5 * (diff^T * cov_inv * diff)
            # Use torch.einsum for batch processing: '...i,...ij,...j->...' performs the matrix multiplication
            quadratic_term = -0.5 * torch.einsum('...i,...ij,...j->...', diff, cov_inv, diff)
            
            # Compute the determinant of the covariance matrix
            cov_det = torch.det(cov_mat)
            
            # Compute the normalization term: 0.5 * log((2 * pi) ^ 2 * det(cov_mat))
            normalization_term = 0.5 * torch.log((2 * torch.pi) ** 2 * cov_det + eps)
            
        # Combine quadratic term and normalization term to compute the log-likelihood
        log_likelihood = quadratic_term #- normalization_term
        
      
        return log_likelihood






    def create_body_length_vect(self):
        # Extract values from the dictionary and convert them to a list
        BPL = list(self.body_lengths.values())
        
        # Convert the list to a tensor
        BPL = torch.tensor(BPL, dtype=self.torch_dtype).unsqueeze(1)
        
        # Repeat each value N times before switching to the next
        repeated_tensor = BPL.repeat_interleave(self.batch_size, dim=0)
        
        # Reshape to [Time*|BPL|, 1]
        final_tensor = repeated_tensor.view(self.batch_size * len(BPL), 1).squeeze()
        
        return final_tensor
    
         
        
        
    def create_batch_indices(self):
        # List to store the batches
        batches = []

#       step_size =  self.batch_size
        step_size = self.batch_size // 2  # Overlap by half the batch size
        
        for start in range(0, self.Time - self.batch_size + 1, step_size):
            batch = list(range(start, start + self.batch_size))
            batches.append(batch)
        return batches


    
    def construct_sample_cost(self):
        #collect estimation heatmaps
        means = self.gaussians_subset[:,2,:, :2].unsqueeze(2).expand(-1,-1,self.N_sample_points,-1)
        #cov_matrices = self.gaussians_subset[:,2,:, 2:].reshape(self.Time, 1, self.n_joints, 2, 2)
        cov_matrices = self.gaussians_subset[:,2,:, 2:].reshape(self.Time, self.n_joints, 2, 2).unsqueeze(2).expand(-1,-1,self.N_sample_points,-1,-1)
        
        
        cmtx1, R1, T1, dist1 = self.decomposed_cam_params[self.GT_camera_IDs[0]]
        cmtx2, R2, T2, dist2 = self.decomposed_cam_params[self.GT_camera_IDs[1]]
        

        self.samples_3d = utils.triangulate_points(self.samples, cmtx1, dist1, R1, T1, cmtx2, dist2, R2, T2)
        self.samples_3d = torch.from_numpy(self.samples_3d).to(self.torch_dtype)
        sample_shape = self.samples_3d.shape
        def cost():
            likelihoods = []
            for ID in self.extrinsic_optimization_IDs:
                params = self.decomposed_cam_params[ID]
                R,T = params[1:3]
                # for n in range(self.N_sample_points):
                    # sample_3d = self.samples_3d[:,:,n,:].squeeze()
                    # points_2d = project_points_torch(sample_3d, *params, torch_dtype=self.torch_dtype, ignore_distortions=self.ignore_distortions)       #project_points_torch(sample_3d, self.decomposed_cam_params[self.estimation_camera_index][0], R, T, decomposed_cam_params[self.estimation_camera_index][-1], torch_dtype=self.torch_dtype)        
                    # likelihoods.append(self.gaussian_likelihood(points_2d, means, cov_matrices, eps=1e-6))
       
                sample_3d = self.samples_3d.reshape(sample_shape[0],-1,sample_shape[-1])

                points_2d = project_points_torch(sample_3d, *params, torch_dtype=self.torch_dtype, ignore_distortions=self.ignore_distortions).reshape(list(sample_shape[:3])+[-1])
                likelihoods.append(self.gaussian_likelihood(points_2d, means, cov_matrices, eps=1e-6, cov_inv=self.cov_invs_subset[ID].unsqueeze(2).expand(-1,-1,self.N_sample_points,-1,-1)))

            mean_likelihood = nan_mean(likelihoods)
            self.extrinsic_param_sample_cost = -mean_likelihood
        self.compute_extrinsic_param_sample_cost = cost


    
    
    def compute_smoothness_cost(self):
        """Computes the smoothness term of the cost function."""
        self.smoothness_costs = []
#        smoothness_cost = 0.0
        for idx in range(2, len(self.indicies)):
            t = self.indicies[idx]
            diff = (self.trajectory[t] - self.trajectory[t - 1]) - (self.trajectory[t - 1] - self.trajectory[t - 2])
            self.smoothness_costs.append(torch.norm(diff) ** 2)
 #           smoothness_cost += torch.norm(diff) ** 2
        self.smoothness_cost = self.lambda_smooth*nan_mean(self.smoothness_costs)
    
    
    def compute_body_length_cost(self):
        BPL = utils.get_body_part_lengths(self.trajectory[self.indicies])
        
        BPL_stacked = torch.hstack([BPL[bl] for bl in self.body_lengths.keys()]).squeeze()
        
        # minimum cost mu: mu = (a^T b) / (||b||^2)
        mu_val = (torch.dot(self.body_length_vect, BPL_stacked)) / torch.dot(BPL_stacked, BPL_stacked)
        
        # Calculate a - lambda * b
        diff = self.body_length_vect - mu_val * BPL_stacked
        
        # Return the squared norm of the difference (divided by |a|^2 for normalization)
        self.body_length_cost = self.lambda_body_length*torch.norm(diff) ** 2 / torch.norm(self.body_length_vect) ** 2
        
    
    def compute_likelihood_cost(self):
        
        self.likelihood_costs = []
        for camera_index, camera_ID in zip(self.camera_indices, self.camera_IDs):
            cam_params = self.decomposed_cam_params[camera_ID]
            trajectory_projection = project_points_torch(self.trajectory, *cam_params, indicies = self.indicies, torch_dtype=self.torch_dtype, ignore_distortions=self.ignore_distortions) if self.n_dims == 3 else self.trajectory[self.indicies]
            cov_inv = self.cov_invs_subset[camera_ID]
            # for idx, T in enumerate(self.indicies):
            #     for joint in range(self.n_joints):
            #             gauss_params = self.gaussians_subset[T, camera_index, joint, :]
            #             mean = gauss_params[:2]
            #             cov_mat = gauss_params[2:].reshape([2,2]) if len(gauss_params[2:])==4 else torch.diag(gauss_params[2:])
                        

            #             if cov_inv is not None:
            #                 CI = cov_inv[camera_index][T][joint]
            #             else: CI = None
            #             likelihood_cost = -self.gaussian_likelihood(trajectory_projection[idx, joint, :], mean, cov_mat, cov_inv = CI)
            #             self.likelihood_costs.append(likelihood_cost)
            if cov_inv is not None:
                CI = cov_inv[self.indicies]
            else: CI = None 
            likelihood_cost = -self.gaussian_likelihood(trajectory_projection, self.gaussians_subset[self.indicies,0,:,:2].squeeze(), self.gaussians_subset[self.indicies,0,:,2:].reshape(len(self.indicies),self.n_joints,2,2), cov_inv=CI)   

            self.likelihood_costs.append(likelihood_cost)
                        
        self.likelihood_cost = nan_mean(self.likelihood_costs)

     
    
    
    def sgd_optimize(self, extrinsic_optimization_IDs = [], optimize_trajectory = True, lr=0.001, betas=(0.9, 0.999), lambda_smooth=1.0, lambda_body_length=1.0, patience=100, tolerance=1e-5, max_iter = 1000, print_frequency = 100, batch_size = None, N_sample_points = 100, GT_camera_IDs = None, ignore_distortions = False, reset_camera_params = False, print_compute_times = False, time_interval = [0,-1], randomize_params = False, use_NN = False):
        
        self.gaussians_subset = self.gaussians[time_interval[0]:time_interval[1]]
        self.cov_invs_subset = {k:self.cov_invs[k][time_interval[0]:time_interval[1]] for k in self.cov_invs}

        self.Time = len(self.gaussians_subset)
        if batch_size is None:
            batch_size = self.Time
            
        #just make batches consistent for now
        self.Time = int(np.floor(self.Time/batch_size)*batch_size)
        self.gaussians_subset = self.gaussians_subset[:self.Time]
        
        
        if reset_camera_params:
            self.decomposed_cam_params = {camera_ID:[cam_param.clone().detach() for cam_param in self.decomposed_cam_params_initial[camera_ID]] for camera_ID in self.decomposed_cam_params_initial}

        self.n_cams = len(self.camera_IDs)
        self.GT_camera_IDs = GT_camera_IDs
        self.ignore_distortions = ignore_distortions
        
        #if we are computing the extrinsics based off of sample points from GT_camera_IDs estimates, ensure that we are setup correctly
        self.learning_extrinsics_from_samples = extrinsic_optimization_IDs is not None and optimize_trajectory is False
        self.extrinsic_optimization_IDs = extrinsic_optimization_IDs
        if self.learning_extrinsics_from_samples:
            if self.GT_camera_IDs is None:
                self.GT_camera_IDs = [idx in self.decomposed_cam_params.keys() for idx in self.GT_camera_IDs if idx not in extrinsic_optimization_IDs]
            assert len(self.extrinsic_optimization_IDs) == 1
            assert len(self.GT_camera_IDs) == 2
            assert self.GT_camera_IDs is not None
            assert min([idx in self.decomposed_cam_params.keys() for idx in self.GT_camera_IDs])
            assert self.extrinsic_optimization_IDs[0] in self.decomposed_cam_params
            

            
        
        self.learnable_params = dict()
        if len(self.extrinsic_optimization_IDs):
            self.learnable_params['cam_params'] = []
            for ID in self.extrinsic_optimization_IDs:
                #convert learnable params to axis-angle representation for ease of learning
                self.decomposed_cam_params_initial[ID][1] = utils.rotation_conversion(self.decomposed_cam_params_initial[ID][1], to_vector=True)
            
                #additionally, make sure that there are no zeros in the initial parameters so that we obtain some signal
                self.decomposed_cam_params[ID][1][self.decomposed_cam_params[ID][1] == 0] = random.random()/10**6
                self.decomposed_cam_params[ID][2][self.decomposed_cam_params[ID][2] == 0] = random.random()/10**6
                
                self.decomposed_cam_params[ID][1].requires_grad_(True)
                self.decomposed_cam_params[ID][2].requires_grad_(True)
                self.learnable_params['cam_params'] += [self.decomposed_cam_params[ID][1],self.decomposed_cam_params[ID][2]]



        # Initialize the trajectory with the provided initial trajectory
        self.trajectory = self.initial_trajectory[time_interval[0]:time_interval[1]].clone().detach()
        if optimize_trajectory:
            if use_NN:
                self.learnable_params['trajectory'] = list(self.simple_nn.parameters())
            else:
                self.trajectory.requires_grad_(True)
                self.learnable_params['trajectory'] = [self.trajectory]


        if randomize_params: 
            for param_class in self.learnable_params:
                if not(param_class == 'cam_params' and reset_camera_params):
                    for param in self.learnable_params[param_class]:
                        torch.nn.init.normal_(param, mean=0.0, std=0.1)
        self.batch_size = batch_size
        
        self.body_length_vect = self.create_body_length_vect()
        self.best_trajectory = None
        self.best_decomposed_cam_params = None
        
        self.lambda_smooth = lambda_smooth
        self.lambda_body_length = lambda_body_length

        
        batch_indicies = self.create_batch_indices()
        
        optimizer = torch.optim.Adam(sum(self.learnable_params.values(), []), lr=lr, betas = betas)
        
        cost_functions = []
        all_costs = {'total_cost':[]}
        if optimize_trajectory:
            all_costs['likelihood_cost'] = []
            cost_functions.append(self.compute_likelihood_cost)
        if lambda_smooth > 0:
            all_costs['smoothness_cost'] = []
            cost_functions.append(self.compute_smoothness_cost)
        if lambda_body_length > 0:
            all_costs['body_length_cost'] = []
            cost_functions.append(self.compute_body_length_cost)
        if self.learning_extrinsics_from_samples:
            self.samples = self.sample_gaussians()
            self.construct_sample_cost()
            all_costs['extrinsic_param_sample_cost'] = []
            cost_functions.append(self.compute_extrinsic_param_sample_cost)
        self.all_costs_total = all_costs.copy()
        best_cost = {cost:float('inf') for cost in all_costs}
        
        no_improvement_count = 0
        iteration=0

        # Initialize cumulative time tracking
        cumulative_cost_times = {cost.__name__: 0.0 for cost in cost_functions}
        cumulative_total_time = 0.0
        
        while no_improvement_count < patience and iteration <= max_iter:
            start_iteration_time = time.time()  # Track start time of the iteration
            
            
            for idx, batch in enumerate(batch_indicies):
                optimizer.zero_grad()
                
                self.indicies = batch
                costs = dict()
                
                if use_NN:
                    temp_trajectory = self.trajectory.detach().clone()
                    temp_trajectory[batch] = self.forward_NN(self.gaussians_subset[batch])
                    self.trajectory = temp_trajectory

                
                # Track batch-specific cost times
                batch_cost_times = {cost.__name__: 0.0 for cost in cost_functions}
                
                # Evaluate each cost and accumulate time taken
                for cost_function in cost_functions:
                    start_time = time.time()
                    cost_function()  # Compute the cost
                    elapsed = time.time() - start_time
                    
                    # Accumulate time taken for this cost function
                    cost_name = cost_function.__name__
                    batch_cost_times[cost_name] = elapsed
                    cumulative_cost_times[cost_name] += elapsed
        
                # Store the costs
                for cost in all_costs:
                    if cost != 'total_cost':
                        costs[cost] = getattr(self, cost)
                        if torch.isnan(costs[cost]):
                            print(f'nan cost for {cost}!')
                            break
                total_cost = torch.sum(torch.stack(list(costs.values())))
                costs['total_cost'] = total_cost
                

                # Perform backward pass
                total_cost.backward()
                
                # Clip gradients to avoid instability
                torch.nn.utils.clip_grad_norm_(sum(self.learnable_params.values(), []), max_norm=1.0)
                
                # Update trajectory using the optimizer
                optimizer.step()
                
                for cost in all_costs:
                    all_costs[cost].append(costs[cost].clone(
                        ).detach())
            
            # Update cumulative total time for the while loop iteration
            cumulative_total_time += time.time() - start_iteration_time
        
            # Print proportional times at specified frequency
            if iteration % print_frequency == 0 and print_compute_times:
                proportional_times = {cost: (time / cumulative_total_time) * 100 for cost, time in cumulative_cost_times.items()}
                print(f"Proportional times for iteration {iteration}: " +
                      ', '.join([f"{cost}: {proportional_times[cost]:.2f}%" for cost in proportional_times]))
                
                # Reset cumulative time tracking
                cumulative_cost_times = {cost.__name__: 0.0 for cost in cost_functions}
                cumulative_total_time = 0.0
        
            # Check for improvement
            for cost in all_costs:
                self.all_costs_total[cost].append(np.mean(all_costs[cost], 0))
            current_costs = {cost:self.all_costs_total[cost][-1] for cost in self.all_costs_total}
            if current_costs['total_cost'] < best_cost['total_cost'] - tolerance:
                best_cost = current_costs
                self.best_trajectory = self.trajectory.clone().detach()
                
                self.best_decomposed_cam_params = dict()
                for k in self.decomposed_cam_params:
                        self.best_decomposed_cam_params[k] = [param.clone().detach() for param in self.decomposed_cam_params[k]]
                        
                no_improvement_count = 0  # Reset no improvement counter
            else:
                no_improvement_count += 1
        
            # Early stopping if no improvement for `patience` iterations
            if no_improvement_count >= patience:
                print(f"Early stopping at iteration {iteration}. " +
                      ', '.join([f"{cost}: {current_costs[cost]:.2e}" for cost in current_costs]))
                break
        
            # Optionally, print the current cost for monitoring
            if iteration % print_frequency == 0:
                print(f"Iteration {iteration}: " +
                      ', '.join([f"{cost}: {current_costs[cost]:.2e}" for cost in current_costs]))
        
            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', type=str, help='Path containing the heatmaps, estimated 3D pose, and log file. Will default to the current path.')
    parser.add_argument('--refinement_types', nargs='+', default=['linear_interpolation'], help='Type(s) of refinement to implement ("linear_interpolation" or "SGD"). Defaults to linear_interpolation')
    parser.add_argument('--recording_log', type=str, help='Path to recording log')
    parser.add_argument('--heatmaps_2d', type=str, help='Path to 2D heatmaps .npy file')
    parser.add_argument('--kpts_2d', type=str, help='Path to 2D keypoints .npy file')
    parser.add_argument('--kpts_3d', type=str, help='Path to 3D keypoints .npy file')
    parser.add_argument('--model', type=str, help='Name of the model used (e.g. coco_base)')
    parser.add_argument('--save_path', type=str, help='Path in which to save estimated pose. Will default to the run path.')
    parser.add_argument('--extrinsic_params_dir', type=str, help='Path to folder in which extrinsic camera parameters exitst. Will default to two levels above the run_path and inside "extrinsic_camera_parameters".')
    parser.add_argument('--intrinsic_params_dir', type=str, help='Path in which to save estimated pose. Will default to ./intrinsic_camera_parameters.')
    parser.add_argument('--refinement_params_yaml', type=str, help='YAML containing parameters for each refinement types. If missing, ./body_part_lengths.yaml will be used if available.')
    parser.add_argument('--body_part_lengths_yaml', type=str, help="YAML subject's body part lengths constructed in accordance with the naming convention defined by utils.POINT_INFO (e.g. left_shoulder_left_elbow: 38). Will default to ./intrinsic_camera_parameters body_part_lengths")
    parser.add_argument('--body_part_lengths_individual_name_yaml', default = 'my_lengths', type=str, help="string specifying the individual (entry in the YAML) to who's body part lengths should be used. Defaults to 'my_lengths'")
    parser.add_argument('--ignore_body_lengths', action='store_true')
    parser.add_argument('--interpolate_before_SGD', action='store_true')
    
    
    args = parser.parse_args()

    if args.run_path is None:
        args.run_path = os.getcwd()
    if args.save_path is None:
        args.save_path = args.run_path


    if args.extrinsic_params_dir is None:
        path = Path(args.run_path)
        args.extrinsic_params_dir = path.parent.parent / "extrinsic_camera_parameters"
    if args.intrinsic_params_dir is None:
        args.intrinsic_params_dir = os.path.join(os.getcwd(), 'intrinsic_camera_parameters')
        

    #define any missing arguments from the recording log
    if args.recording_log is not None:
        with open(args.recording_log) as f:
            log = yaml.safe_load(f)
    elif os.path.exists(os.path.join(args.run_path, 'recording_log.yaml')):
        with open(os.path.join(args.run_path, 'recording_log.yaml')) as f:
            log = yaml.safe_load(f)        
    
    args.recording_log = log
    for arg_name, arg_value in vars(args).items():
        if arg_value is None and arg_name in log:
            setattr(args, arg_name, log[arg_name])
    
    
    
    #load data
    kpts_3d = utils.load_if_exists(args.kpts_3d) 
    kpts_2d = utils.load_if_exists(args.kpts_2d)
    heatmaps = utils.load_if_exists(args.heatmaps_2d)
    heatmaps = torch.tensor(heatmaps)
    model_name = args.model
    save_path = args.save_path    
    extrinsic_params_dir = args.extrinsic_params_dir
    intrinsic_params_dir = args.intrinsic_params_dir
    refinement_types = set(args.refinement_types)
    refinement_params_yaml = args.refinement_params_yaml
    interpolate_before_SGD = args.interpolate_before_SGD
    
    
    
    
    
    #Load function arguments (from yaml file if present)
    params = utils.load_config(refinement_params_yaml)
    

    #perfom linear intoerpolation to be saved or used as initialization for SGD
    kwargs = utils.prepare_kwargs(linear_interpolation, params.get("linear_interpolation"))
    
    kpts_3d_interpolation = linear_interpolation(kpts_3d, **kwargs)
    if 'linear_interpolation' in refinement_types:
        interpolation_save_path = os.path.join(save_path, 'kpts_3d_linear_interpolation.npy')
        print(f'saving linear interpolation at {interpolation_save_path}')
        np.save(interpolation_save_path, kpts_3d_interpolation)
        refinement_types.remove('linear_interpolation')
        
        
        
    if 'SGD' in refinement_types:
        
        
        camera_names_pickle_file = os.path.join(extrinsic_params_dir, 'camera_names.pkl')
        with open(camera_names_pickle_file, "rb") as f:
            cameras, origin_camera = pk.load(f)

        Ps = {i:None for i in cameras.keys()}
        decomposed_cam_params = {i:None for i in cameras.keys()}

        for i in cameras.keys():
            Ps[i], decomposed_cam_params[i] = utils.get_params_from_name(cameras[i], intrinsic_params_dir=intrinsic_params_dir, extrinsic_params_dir=extrinsic_params_dir)
        print(f'PARAMS {decomposed_cam_params}')

        

        #load body part lengths if desired
        if not args.ignore_body_lengths:
            if args.body_part_lengths_yaml is None and os.path.exists('./body_part_lengths.yaml'):                    
                args.body_part_lengths_yaml = './body_part_lengths.yaml'
            if args.body_part_lengths_yaml is not None:
                with open(args.body_part_lengths_yaml, "r") as f:
                    body_part_lengths = yaml.safe_load(f)
                my_lengths = body_part_lengths[args.body_part_lengths_individual_name_yaml]
            else:
                my_lengths = None

        if interpolate_before_SGD:
            kpts_3d = kpts_3d_interpolation
        optimize_pose = Optimized_3d_Pose_Estimation(heatmaps, kpts_3d, decomposed_cam_params_initial=decomposed_cam_params, body_lengths = my_lengths)

        #collect kwargs for SGD
        kwargs = utils.prepare_kwargs(optimize_pose.sgd_optimize, params.get("SGD"))
        optimize_pose.sgd_optimize(**kwargs)
        if my_lengths is not None:
                
                
            # #compute the average reporjection error    
            # reprojects = []
            # for i in decomposed_cam_params.keys():
            #     K, R, T, dist_coeffs = decomposed_cam_params[i]
            #     reprojects.append(utils.project_points(optimize_pose.initial_trajectory,K, R, T, dist_coeffs))
                
            # reprojects = np.array(reprojects)
            # reprojects = np.moveaxis(reprojects, 0, -1)
            # print(f'average initial reprojection error: {np.mean(np.abs(reprojects - kpts_2d[:,:,:2,:]))}')
    
            # reprojects = []
            # for i in decomposed_cam_params.keys():
            #     K, R, T, dist_coeffs = decomposed_cam_params[i]
            #     reprojects.append(utils.project_points(optimize_pose.best_trajectory,K, R, T, dist_coeffs))
                
            # reprojects = np.array(reprojects)
            # reprojects = np.moveaxis(reprojects, 0, -1)
            # print(f'average final reprojection error: {np.mean(np.abs(reprojects - kpts_2d[:,:,:2,:]))}')
    

            
            print("mean and standardized error of initial trajectory's body part lengths")
            body_part_lengths = utils.get_body_part_lengths(optimize_pose.initial_trajectory)
            for bp in my_lengths:
                print('; '.join([bp, str(torch.mean(body_part_lengths[bp])), str(torch.std(body_part_lengths[bp]))]))

            print("mean and standardized error of the estimated trajectory's' body part lengths")
            body_part_lengths = utils.get_body_part_lengths(optimize_pose.best_trajectory)
            for bp in my_lengths:
                print('; '.join([bp, str(torch.mean(body_part_lengths[bp])), str(torch.std(body_part_lengths[bp]))]))

        
        SGD_save_path = os.path.join(save_path, 'kpts_3d_SGD.npy')
        print(f'saving SGD at {SGD_save_path}')    
        np.save(SGD_save_path,np.array(optimize_pose.best_trajectory))
        
        
        refinement_types.remove('SGD')
    
    
    