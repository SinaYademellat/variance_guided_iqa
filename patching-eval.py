import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
import csv
import argparse
from PIL import Image
from torchvision import transforms
import math


class WeightedPatchIDFIQA(nn.Module):
    """
    Weighted patch-based IDFIQA metric.
    Uses a separate weighting layer to compute spatial importance weights,
    then calculates IDFIQA on patches and returns the weighted mean.
    """
    def __init__(self, feature_extractor, normalize,
                 device=None, percent_features_to_keep=0.5, window_size=2,
                 patch_size=64):
        """
        Args:
            feature_extractor: Single extractor returning both 'features' and 'weights'.
            normalize: Normalization transform for input images.
            device: Torch device.
            percent_features_to_keep: Percentage of channels to keep based on variance.
            window_size: Window size for SSIM-like calculation.
            patch_size: Size of patches to divide features into.
        """
        super(WeightedPatchIDFIQA, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.window_size = window_size
        self.xi = 1e-8
        self.percent_features_to_keep = percent_features_to_keep
        self.patch_size = patch_size

        self.feature_extractor = feature_extractor.to(self.device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.normalize = normalize

    def _extract_all(self, img):
        """Extract both features and weight map in a single forward pass."""
        img_norm = self.normalize(img)
        outputs = self.feature_extractor(img_norm)
        return outputs['features'], outputs['weights']

    def _compute_weight_map(self, weight_features):
        """Compute weight map using L2-norm across channels."""
        weight_map = torch.norm(weight_features, p=2, dim=1)  # (n, h, w)
        return weight_map

    def _compute_gram_matrix(self, feature_map):
        n, c, h, w = feature_map.size()
        features_reshaped = feature_map.view(n, c, h * w)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        return gram / (h * w)

    def _compute_patch_score(self, patch_ref, patch_dist):
        """Compute IDFIQA score for a single patch pair."""
        n, c, h, w = patch_ref.shape

        # Select top K% channels based on reference variance
        ref_variances = torch.var(patch_ref, dim=(2, 3), unbiased=False)
        num_channels_to_keep = max(1, int(c * self.percent_features_to_keep))
        _, top_indices = torch.topk(ref_variances, num_channels_to_keep, dim=1)

        top_indices_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        selected_ref = torch.gather(patch_ref, 1, top_indices_expanded)
        selected_dist = torch.gather(patch_dist, 1, top_indices_expanded)

        # Compute gram matrices
        gram_ref = self._compute_gram_matrix(selected_ref)
        gram_dist = self._compute_gram_matrix(selected_dist)

        # SSIM-like calculation on gram matrices
        gram_ref_unf = F.unfold(gram_ref.unsqueeze(1), kernel_size=self.window_size, stride=1, padding=0)
        gram_dist_unf = F.unfold(gram_dist.unsqueeze(1), kernel_size=self.window_size, stride=1, padding=0)
        gram_ref_unf = gram_ref_unf.transpose(1, 2)
        gram_dist_unf = gram_dist_unf.transpose(1, 2)

        var_ref = torch.var(gram_ref_unf, dim=2, unbiased=False)
        var_dist = torch.var(gram_dist_unf, dim=2, unbiased=False)
        mean_ref = torch.mean(gram_ref_unf, dim=2, keepdim=True)
        mean_dist = torch.mean(gram_dist_unf, dim=2, keepdim=True)
        covar = torch.mean((gram_ref_unf - mean_ref) * (gram_dist_unf - mean_dist), dim=2)

        local_scores = (2 * covar + self.xi) / (var_ref + var_dist + self.xi)
        score = torch.mean(local_scores, dim=1)
        return score

    def forward(self, ref_img, dist_img):
        ref_img = ref_img.to(self.device)
        dist_img = dist_img.to(self.device)

        # Extract features and weight map in single forward passes
        features_ref, weight_features_ref = self._extract_all(ref_img)
        features_dist, _ = self._extract_all(dist_img)

        # Compute weight map from weight features
        weight_map = self._compute_weight_map(weight_features_ref)  # (n, h_w, w_w)
        _, _, h_feat, w_feat = features_ref.shape
        weight_map = F.interpolate(weight_map.unsqueeze(1), size=(h_feat, w_feat),
                                   mode='bilinear', align_corners=False).squeeze(1)  # (n, h_feat, w_feat)

        n, c, h, w = features_ref.shape

        # Calculate number of patches
        num_patches_h = max(1, h // self.patch_size)
        num_patches_w = max(1, w // self.patch_size)

        # Adjust patch size to evenly divide the feature map
        actual_patch_h = h // num_patches_h
        actual_patch_w = w // num_patches_w

        scores = []
        weights = []

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                h_start = i * actual_patch_h
                h_end = (i + 1) * actual_patch_h if i < num_patches_h - 1 else h
                w_start = j * actual_patch_w
                w_end = (j + 1) * actual_patch_w if j < num_patches_w - 1 else w

                patch_ref = features_ref[:, :, h_start:h_end, w_start:w_end]
                patch_dist = features_dist[:, :, h_start:h_end, w_start:w_end]
                patch_weight = weight_map[:, h_start:h_end, w_start:w_end]

                # Skip patches that are too small for the window size
                if patch_ref.shape[2] < self.window_size or patch_ref.shape[3] < self.window_size:
                    continue

                # Compute patch score
                patch_score = self._compute_patch_score(patch_ref, patch_dist)

                # Weight is the max value of weight map in this patch
                patch_weight_scalar = torch.max(patch_weight.reshape(n, -1), dim=1)[0]

                scores.append(patch_score)
                weights.append(patch_weight_scalar)

        if len(scores) == 0:
            # Fallback: compute score on entire feature map
            return self._compute_patch_score(features_ref, features_dist)

        scores = torch.stack(scores, dim=1)  # (n, num_patches)
        weights = torch.stack(weights, dim=1)  # (n, num_patches)

        # Weighted mean
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.xi)
        weighted_score = (scores * weights).sum(dim=1)

        return weighted_score


def get_dual_feature_extractor(backbone_name, feature_layer, weight_layer):
    """Creates a single feature extractor that returns both feature and weight maps.

    Args:
        backbone_name: Name of the backbone model ('vgg16' or 'efficientnet_b4')
        feature_layer: Layer for IDFIQA features
        weight_layer: Layer for weight map

    Returns:
        tuple: (feature_extractor, normalize)
    """
    if backbone_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        return_nodes = {feature_layer: 'features', weight_layer: 'weights'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        normalize = weights.transforms()
        return feature_extractor, normalize

    elif backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        return_nodes = {feature_layer: 'features', weight_layer: 'weights'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        normalize = weights.transforms()
        return feature_extractor, normalize

    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported.")


class AIC4EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images, structured as 'source' and 'distorted'.
        """
        self.root_dir = root_dir
        self.source_dir = os.path.join(root_dir, 'source')
        self.distorted_dir = os.path.join(root_dir, 'distorted')

        self.image_pairs = []
        # List all source images
        source_images = sorted([f for f in os.listdir(self.source_dir) if f.endswith('.png')])

        # Iterate through each source image and find its distorted counterparts
        for src_img_name in source_images:
            src_img_path = os.path.join(self.source_dir, src_img_name)
            # The distorted images for a source image are in a subfolder named after the source image
            distorted_subfolder = os.path.join(self.distorted_dir, src_img_name)

            if os.path.isdir(distorted_subfolder):
                # List all distorted images in the subfolder
                distorted_images = sorted([f for f in os.listdir(distorted_subfolder) if f.endswith('.png')])
                for dis_img_name in distorted_images:
                    dis_img_path = os.path.join(distorted_subfolder, dis_img_name)
                    self.image_pairs.append((src_img_path, dis_img_path, src_img_name, dis_img_name))

        # Define transformations (resize and convert to tensor, normalize later if needed by model)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        src_img_path, dis_img_path, src_img_name, dis_img_name = self.image_pairs[idx]

        # Load images
        ref_img = Image.open(src_img_path).convert('RGB')
        dist_img = Image.open(dis_img_path).convert('RGB')

        # Apply transformations
        ref_img_tensor = self.transform(ref_img)
        dist_img_tensor = self.transform(dist_img)

        return {
            'ref_img': ref_img_tensor,
            'dis_img': dist_img_tensor,
            'ref_img_name': src_img_name,
            'dis_img_name': dis_img_name
        }

def evaluate_single_pair(model, ref_path, dist_path, device):
    """
    Evaluate a single pair of reference and distorted images.
    
    Args:
        model: WeightedPatchIDFIQA model
        ref_path: Path to reference image
        dist_path: Path to distorted image
        device: torch device
        
    Returns:
        float: WeightedPatchIDFIQA score
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load images
    ref_img = Image.open(ref_path).convert('RGB')
    dist_img = Image.open(dist_path).convert('RGB')
    
    # Apply transformations and add batch dimension
    ref_img_tensor = transform(ref_img).unsqueeze(0).to(device)
    dist_img_tensor = transform(dist_img).unsqueeze(0).to(device)
    
    # Calculate score
    with torch.no_grad():
        score = model(ref_img_tensor, dist_img_tensor).item()
    
    return score

def JND(x):
    b = 1.0
    a = 15.0
    return a * max(0, b - x)

def main():
    parser = argparse.ArgumentParser(description='Evaluate image quality using weighted patch-based IDFIQA')
    parser.add_argument('--mode', choices=['dataset', 'single'], default='single',
                        help='Evaluation mode: "dataset" for full dataset or "single" for single pair')
    parser.add_argument('--root-dir', type=str, default='aic4-evaluation',
                        help='Root directory of dataset (for dataset mode)')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Output CSV file path (for dataset mode)')
    parser.add_argument('--ref-img', type=str,
                        help='Path to reference image (for single mode)')
    parser.add_argument('--dist-img', type=str,
                        help='Path to distorted image (for single mode)')
    parser.add_argument('--percent-features', type=float, default=0.7,
                        help='Percentage of feature maps to keep (0.0 to 1.0), default=0.7')
    parser.add_argument('--patch-size', type=int, default=8,
                        help='Size of patches in feature space, default=8')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'single':
        if not args.ref_img or not args.dist_img:
            parser.error("--ref-img and --dist-img are required for single mode")
        if not os.path.exists(args.ref_img):
            parser.error(f"Reference image not found: {args.ref_img}")
        if not os.path.exists(args.dist_img):
            parser.error(f"Distorted image not found: {args.dist_img}")
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = "efficientnet_b4"
    feature_layer = 'features.5.1.block.1'
    weight_layer = 'features.6.5.block.1'
    
    feature_extractor, normalize = get_dual_feature_extractor(backbone, feature_layer, weight_layer)
    
    model = WeightedPatchIDFIQA(
        feature_extractor=feature_extractor,
        normalize=normalize,
        device=device,
        window_size=4,
        percent_features_to_keep=args.percent_features,
        patch_size=args.patch_size
    )
    
    model.eval()
    
    if args.mode == 'single':
        # Single pair evaluation
        score = evaluate_single_pair(model, args.ref_img, args.dist_img, device)
        mapped_score = JND(score)
        print(f"WeightedPatchIDFIQA Score: {mapped_score:.6f}")
        print(f"Reference: {args.ref_img}")
        print(f"Distorted: {args.dist_img}")
    else:
        # Dataset evaluation
        if not os.path.exists(args.root_dir):
            parser.error(f"Root directory not found: {args.root_dir}")
        
        eval_dataset = AIC4EvaluationDataset(args.root_dir)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, 
                                    shuffle=False, num_workers=args.num_workers)
        
        with open(args.output, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['ref_img_name', 'dis_img_name', 'quality_score', 'jnd_mapped'])
            csvfile.flush()
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), 
                                desc="Calculating WeightedPatchIDFIQA"):
                    ref_img = batch['ref_img'].to(device)
                    dist_img = batch['dis_img'].to(device)
                    ref_img_name = batch['ref_img_name'][0]
                    dis_img_name = batch['dis_img_name'][0]
                    
                    score = model(ref_img, dist_img).item()
                    
                    csv_writer.writerow([ref_img_name, dis_img_name, score, JND(score)])
                    csvfile.flush()
        
        print(f"WeightedPatchIDFIQA results saved to {args.output}")

if __name__ == '__main__':
    main()
