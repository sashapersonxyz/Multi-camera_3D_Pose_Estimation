import time

import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
import torch

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    

def bresenham_line(x0, y0, x1, y1):
    """Return coordinates of a line between (x0, y0) and (x1, y1) using Bresenham."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    rr, cc = zip(*points)
    return np.array(rr), np.array(cc)

def zero_out_slopes(heatmaps):
    """
    For each heatmap in a batch, zero out pixels that come after
    an increase along rays from the max pixel to all boundary pixels.
    """
    n, H, W = heatmaps.shape
    out = heatmaps.copy()

    for idx in range(n):
        hm = out[idx]
        i_max, j_max = np.unravel_index(np.argmax(hm), hm.shape)

        # Boundary pixels
        boundary_pixels = [(0, j) for j in range(W)] + \
                          [(H-1, j) for j in range(W)] + \
                          [(i, 0) for i in range(H)] + \
                          [(i, W-1) for i in range(H)]
        boundary_pixels = list(set(boundary_pixels))

        for (i_b, j_b) in boundary_pixels:
            rr, cc = bresenham_line(i_max, j_max, i_b, j_b)
            vals = hm[rr, cc]

            # Walk along values and zero after slope up
            last_val = vals[0]
            zero_mode = False
            for k in range(1, len(vals)):
                if not zero_mode and vals[k] > last_val:
                    zero_mode = True
                if zero_mode:
                    hm[rr[k], cc[k]] = 0
                last_val = vals[k]

    return out

def normalize_per_sample(arr):
    arr = arr.astype(float)  # make sure we’re working with floats
    min_vals = arr.min(axis=(1,2), keepdims=True)
    max_vals = arr.max(axis=(1,2), keepdims=True)
    return (arr - min_vals) / (max_vals - min_vals + 1e-12)

class PoseEstimator:
    def __init__(self, det_config, det_checkpoint, pose_config, pose_checkpoint,
                 device='cpu', det_cat_id=0, bbox_thr=0.3, nms_thr=0.3, using_detector = True):
        """
        Initialize the PoseEstimator class with detector and pose estimator models.
        
        Args:
        - det_config: Config file for detection.
        - det_checkpoint: Checkpoint file for detection.
        - pose_config: Config file for pose.
        - pose_checkpoint: Checkpoint file for pose.
        - device (str): Device used for inference (default: 'cpu').
        - det_cat_id (int): Category id for bounding box detection model (default: 0).
        - bbox_thr (float): Bounding box score threshold (default: 0.3).
        - nms_thr (float): IoU threshold for bounding box NMS (default: 0.3).
        """
        # Build detector
        self.detector = init_detector(det_config, det_checkpoint, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        # self.detector = mmdet.apis.DetInferencer('rtmdet_tiny_8xb32-300e_coco', device = device)

        # Build pose estimator
        self.pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=True)))
        )
        self.device = device
        self.det_cat_id = det_cat_id
        self.bbox_thr = bbox_thr
        self.using_detector = using_detector
    @staticmethod
    def get_heatmap_means_stds(heatmaps):
        all_means, all_stds = [], []
        
        # If the input is a list, apply the function recursively
        if isinstance(heatmaps, list):
            for heatmap in heatmaps:
                means, stds = PoseEstimator.get_heatmap_means_stds(heatmap)
                all_means.append(means)
                all_stds.append(stds)
            return all_means, all_stds

        # Create coordinate grids for height (y) and width (x)
        y_grid = torch.arange(heatmaps.shape[1]).view(heatmaps.shape[1], 1).expand(heatmaps.shape[1], heatmaps.shape[2]).float()
        x_grid = torch.arange(heatmaps.shape[2]).view(1, heatmaps.shape[2]).expand(heatmaps.shape[1], heatmaps.shape[2]).float()

        means, stds = [], []

        # Loop through each heatmap
        for heatmap in heatmaps:
            heatmap_sum = heatmap.sum()

            # Handle empty or zero-sum heatmaps by skipping or adding default values
            if heatmap_sum == 0:
                means.append((0.0, 0.0))
                stds.append((0.0, 0.0))
                continue

            # Normalize the heatmap
            normalized_heatmap = heatmap / heatmap_sum

            # Calculate the mean coordinates
            mean_x = (x_grid * normalized_heatmap).sum()
            mean_y = (y_grid * normalized_heatmap).sum()

            # Calculate the variance (standard deviation is the square root of variance)
            var_x = ((x_grid - mean_x) ** 2 * normalized_heatmap).sum()
            var_y = ((y_grid - mean_y) ** 2 * normalized_heatmap).sum()

            # Replace NaN stds with zero (in case of extremely flat distributions)
            std_x = var_x.sqrt().item() if not torch.isnan(var_x.sqrt()) else 0.0
            std_y = var_y.sqrt().item() if not torch.isnan(var_y.sqrt()) else 0.0

            # Store the results
            means.append((mean_x.item(), mean_y.item()))
            stds.append((std_x, std_y))

        return means, stds

    def get_heatmap_means_cov(self, heatmaps):
        
        #ensure there are no negatives (otherwise cov computations are all wrong). Note, we also zero near negatives since otherwise the mean and cov will be skewed by the edges of the frame
        heatmaps[heatmaps<0.01] = 0
        #heatmaps = zero_out_slopes(heatmaps)
        # #actually, instead renormalize to maintain information
        #heatmaps = normalize_per_sample(heatmaps)
        
        all_means_covs = []
    
        # If the input is a list, apply the function recursively
        if isinstance(heatmaps, list):
            for heatmap in heatmaps:
                means_covs = self.get_heatmap_means_cov(heatmap)
                all_means_covs.append(means_covs)
            return np.array(all_means_covs)
    
        # Create coordinate grids for height (y) and width (x)
        y_grid = torch.arange(heatmaps.shape[1]).view(heatmaps.shape[1], 1).expand(heatmaps.shape[1], heatmaps.shape[2]).float()
        x_grid = torch.arange(heatmaps.shape[2]).view(1, heatmaps.shape[2]).expand(heatmaps.shape[1], heatmaps.shape[2]).float()
    
        means_covs = []
    
        # Loop through each heatmap
        for heatmap in heatmaps:
            heatmap_sum = heatmap.sum()
    
            # Handle empty or zero-sum heatmaps by skipping or adding default values
            if heatmap_sum == 0:
                means_covs.append(np.zeros(6))  # Default value for mean and covariance
                continue
    
            # Normalize the heatmap
            normalized_heatmap = heatmap / heatmap_sum
    
            # Calculate the mean coordinates
            mean_x = (x_grid * normalized_heatmap).sum()
            mean_y = (y_grid * normalized_heatmap).sum()
    
            # Calculate the variances and covariances
            var_x = ((x_grid - mean_x) ** 2 * normalized_heatmap).sum()
            var_y = ((y_grid - mean_y) ** 2 * normalized_heatmap).sum()
            cov_xy = ((x_grid - mean_x) * (y_grid - mean_y) * normalized_heatmap).sum()
    
            # Create the combined mean and covariance array
            combined = np.array([mean_x.item(), mean_y.item(),
                                 var_x.item(), cov_xy.item(),
                                 cov_xy.item(), var_y.item()])
    
            # Store the result
            means_covs.append(combined)
    
        return np.array(means_covs)
    
        
            
    


    def predict(self, input_file, return_full_heatmaps = False):
        """
        Predict the keypoints and heatmaps for the input image/video file.

        Args:
        - input_file: Image or video file path.

        Returns:
        - pred_instances: Predicted keypoints.
        - heatmaps: Means and standard deviations of the heatmaps.
        """
        # Predict bounding boxes
        if self.using_detector:
            t0=time.time()
            det_result = inference_detector(self.detector, input_file)
            # det_result = self.detector(input_file)
            t1=time.time()
            #print(f'detection time {t1-t0}')

            pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id,
                                       pred_instance.scores > self.bbox_thr)]
        # Get only one bbox (assuming only one person in video), also remove the score from the result
        try:
            bboxes = bboxes[0, :4]
            bboxes = np.expand_dims(bboxes, 0)
        except:
            bboxes = None
        t0=time.time()
        # Predict keypoints
        pose_results = inference_topdown(self.pose_estimator, input_file, bboxes)
        data_samples = merge_data_samples(pose_results)
        t1=time.time()
        #print(f'estimation time {t1-t0}')

        heatmaps = data_samples.get('_pred_heatmaps', None)
        heatmaps = list(heatmaps.all_values())[0]

        # #get heatmaps as sprase tensors otherwise we'll run out of space real fast
        # heatmaps = torch.from_numpy(heatmaps).to_sparse()
        
        #actually, this is still bad. instead just return means and stds
        #heatmaps = self.get_heatmap_means_stds(heatmaps)
        if not return_full_heatmaps:
            heatmaps = self.get_heatmap_means_cov(heatmaps)
 
    
 
        # Return instances and heatmaps
        return data_samples.get('pred_instances', None), heatmaps









