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

class IDFIQA(nn.Module):
    """
    A backbone-agnostic implementation of the IDFIQA metric.
    It accepts a pre-initialized feature extractor and its corresponding
    normalization function.
    MODIFIED: This version selects a percentage of feature maps based on
    the reference image's channel-wise variance before calculation.
    """
    def __init__(self, feature_extractor, normalize, feature_node_key='features', lite_version=False, device=None, percent_features_to_keep=0.5, window_size=2):
        """
        Initializes the IDFIQA model.

        Args:
            feature_extractor (nn.Module): A pre-initialized feature extraction model.
            normalize (callable): A function or transform to normalize input images for the feature extractor.
            feature_node_key (str): The key to access the feature tensor if the extractor returns a dict.
            lite_version (bool): If True, uses the faster IDFIQA-Lite version.
            device (torch.device, optional): The device to run the model on.
            percent_features_to_keep (float): The percentage of feature maps to keep based on variance (0.0 to 1.0).
            window_size (int): The size of the window for SSIM calculation in the non-lite version.
        """
        super(IDFIQA, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.lite_version = lite_version
        self.window_size = window_size
        self.xi = 1e-8
        self.percent_features_to_keep = percent_features_to_keep

        # Store the supplied feature extractor and normalization function
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.normalize = normalize
        self.feature_node_key = feature_node_key

    def _get_deep_features(self, img):
        img_norm = self.normalize(img)
        features = self.feature_extractor(img_norm)
        # Handle dictionary output from create_feature_extractor
        if isinstance(features, dict):
            return features[self.feature_node_key]
        return features

    def _compute_gram_matrix(self, feature_map):
        n, c, h, w = feature_map.size()
        features_reshaped = feature_map.view(n, c, h * w)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        return gram / (h * w)

    def forward(self, ref_img, dist_img):
        # 1. Extract all features first
        features_ref = self._get_deep_features(ref_img.to(self.device))
        features_dist = self._get_deep_features(dist_img.to(self.device))

        # 2. Select top K% of features based on reference image variance
        n, c, h_ref, w_ref = features_ref.shape

        # Calculate variance for each channel across spatial dimensions
        ref_variances = torch.var(features_ref, dim=(2, 3), unbiased=False)

        # Determine the number of channels to keep based on the percentage
        num_channels_to_keep = int(c * self.percent_features_to_keep)
        if num_channels_to_keep == 0 and c > 0:
             num_channels_to_keep = 1 # Ensure at least one channel if features exist


        # Get the indices of the channels with the highest variance
        _, top_indices = torch.topk(ref_variances, num_channels_to_keep, dim=1)

        # 3. Filter both reference and distorted features using these indices
        # Expand indices to match the feature map dimensions for gathering
        top_indices_ref = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_ref, w_ref)
        selected_features_ref = torch.gather(features_ref, 1, top_indices_ref)

        _, _, h_dist, w_dist = features_dist.shape
        top_indices_dist = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_dist, w_dist)
        selected_features_dist = torch.gather(features_dist, 1, top_indices_dist)


        # 4. Proceed with IDFIQA calculation on the selected features
        gram_ref = self._compute_gram_matrix(selected_features_ref)
        gram_dist = self._compute_gram_matrix(selected_features_dist)

        if self.lite_version:
            var_ref = torch.var(gram_ref, dim=(1, 2), unbiased=False)
            var_dist = torch.var(gram_dist, dim=(1, 2), unbiased=False)
            mean_ref = torch.mean(gram_ref, dim=(1, 2), keepdim=True)
            mean_dist = torch.mean(gram_dist, dim=(1, 2), keepdim=True)
            covar = torch.mean((gram_ref - mean_ref) * (gram_dist - mean_dist), dim=(1, 2))
            score = (2 * covar + self.xi) / (var_ref + var_dist + self.xi)
        else:
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

def get_feature_extractor(backbone_name, layer_name=None):
    """Creates a feature extractor and its normalization function based on the backbone name.
    
    Args:
        backbone_name: Name of the backbone model ('vgg16' or 'efficientnet_b4')
        layer_name: Specific layer to extract features from (optional, uses defaults if not provided)
        
    Returns:
        tuple: (feature_extractor, normalize, feature_node_key)
    """
    if backbone_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        # Use 'features.23' for conv5_1 layer in VGG16 as default
        default_layer = 'features.23'
        layer = layer_name if layer_name is not None else default_layer
        return_nodes = {layer: 'features'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        normalize = weights.transforms()
        return feature_extractor, normalize, 'features'

    elif backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        # Default layer for EfficientNet-B4
        default_layer = 'features.5.1.block.1'
        layer = layer_name if layer_name is not None else default_layer
        return_nodes = {layer: 'features'}
        feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
        normalize = weights.transforms()
        return feature_extractor, normalize, 'features'

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
        model: IDFIQA model
        ref_path: Path to reference image
        dist_path: Path to distorted image
        device: torch device
        
    Returns:
        float: IDFIQA score
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
    parser = argparse.ArgumentParser(description='Evaluate image quality')
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
    layer = 'features.5.1.block.1'
    feature_extractor, normalize, feature_node_key = get_feature_extractor(backbone, layer)
    
    model = IDFIQA(
        feature_extractor=feature_extractor,
        normalize=normalize,
        feature_node_key=feature_node_key,
        lite_version=False,
        device=device,
        window_size=4,
        percent_features_to_keep=args.percent_features
    )
    
    model.eval()
    
    if args.mode == 'single':
        # Single pair evaluation
        score = evaluate_single_pair(model, args.ref_img, args.dist_img, device)
        mapped_score = JND(score)
        print(f"IDFIQA Score: {mapped_score:.6f}")
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
                                desc="Calculating IDFIQA"):
                    ref_img = batch['ref_img'].to(device)
                    dist_img = batch['dis_img'].to(device)
                    ref_img_name = batch['ref_img_name'][0]
                    dis_img_name = batch['dis_img_name'][0]
                    
                    IDFIQA_score = model(ref_img, dist_img).item()
                    
                    csv_writer.writerow([ref_img_name, dis_img_name, IDFIQA_score, JND(IDFIQA_score)])
                    csvfile.flush()
        
        print(f"IDFIQA results saved to {args.output}")

if __name__ == '__main__':
    main()

