import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from captum.attr import IntegratedGradients

# Local imports
from modules.model import Transformer_DAPT


class FeatureMapper:
    """
    A utility class to map between feature indices and their names.
    This is useful for model interpretability, allowing users to identify
    which features are active for a given sample.

    Args:
        feature_names (list): List of feature names.
    """
    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.index_to_feature = {i: name for i, name in enumerate(feature_names)}
        self.feature_to_index = {name: i for i, name in enumerate(feature_names)}
        self.padding_idx = len(feature_names)
    
    def get_feature_names(self, indices):
        """
        Convert feature indices (excluding padding) back to feature names.
        
        Args:
            indices (list or array): List of feature indices.
        
        Returns:
            list: Corresponding feature names.
        """
        return [self.index_to_feature[idx] for idx in indices if idx != self.padding_idx]
    
    def get_sample_features(self, model_input, attention_mask):
        """
        Retrieve active (non-padded) features for a given sample.

        Args:
            model_input (Tensor): Input tensor representing features.
            attention_mask (Tensor): Attention mask tensor.

        Returns:
            list: Active feature names.
        """
        active_indices = model_input[attention_mask.bool()].cpu().numpy()
        return self.get_feature_names(active_indices)


def load_trained_model(model_path, config):
    """
    Load a pre-trained model from the specified checkpoint.

    Args:
        model_path (str): Path to the model checkpoint file.
        config (dict): Configuration dictionary for the model.

    Returns:
        nn.Module: Loaded model in evaluation mode.
    """
    model = Transformer_DAPT(config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def calculate_ig_importance(model, x_test, mask_test, config, device, batch_size=32, seed=44):
    """
    Calculate feature importance using Integrated Gradients.

    Args:
        model (nn.Module): Trained model for which to calculate importance.
        x_test (Tensor): Test data input tensor.
        mask_test (Tensor): Attention mask tensor.
        config (dict): Configuration dictionary.
        device (torch.device): Device to use for calculations.
        batch_size (int): Batch size for processing. Default: 32.
        seed (int): Seed for reproducibility. Default: 44.

    Returns:
        tuple: Aggregated importance scores and results directory.
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    
    # Create results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "IG_results")
    os.makedirs(results_dir, exist_ok=True)
    
    def forward_func(embeddings, mask):
        """
        Forward function for Integrated Gradients.

        Args:
            embeddings (Tensor): Embeddings tensor.
            mask (Tensor): Attention mask tensor.

        Returns:
            Tensor: Binary logits from the model.
        """
        hidden_state = model.feature_encoder(embeddings.view(-1, embeddings.size(-1)))
        stacked_features = hidden_state.view(embeddings.size(0), embeddings.size(1), -1)
        
        attn_output, _ = model.feature_attention(
            stacked_features, stacked_features, stacked_features,
            key_padding_mask=(mask == 0)
        )
        
        hidden_state = attn_output.reshape(attn_output.size(0), -1)
        hidden_state = model.dropout(hidden_state)
        _, _, binary_logits = model.cls(hidden_state)
        return binary_logits

    # Initialize Integrated Gradients
    ig = IntegratedGradients(forward_func)

    n_batches = (len(x_test) + batch_size - 1) // batch_size
    aggregated_importance = {}

    print(f"\nProcessing {len(x_test)} samples in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(x_test))

        try:
            # Get batch
            input_tensor = x_test[start_idx:end_idx].to(device)
            mask_tensor = mask_test[start_idx:end_idx].to(device)
            
            # Generate baselines
            baseline_input = torch.full_like(input_tensor, config.padding_idx)
            input_embeddings = model.embeddings(input_tensor).detach().requires_grad_()
            baseline_embeddings = model.embeddings(baseline_input.to(device)).detach()

            # Integrated Gradients
            attributions, _ = ig.attribute(
                inputs=input_embeddings,
                baselines=baseline_embeddings,
                additional_forward_args=(mask_tensor,),
                n_steps=50,
                return_convergence_delta=True
            )

            attributions = attributions.sum(dim=2).cpu().detach().numpy()
            feature_indices = input_tensor.cpu().numpy()
            
            # Aggregate importance scores
            for sample_idx in range(end_idx - start_idx):
                active_features = config.feature_mapper.get_feature_names(feature_indices[sample_idx])
                sample_attributions = attributions[sample_idx]
                
                for feat, attr in zip(active_features, sample_attributions[:len(active_features)]):
                    if feat not in aggregated_importance:
                        aggregated_importance[feat] = []
                    aggregated_importance[feat].append(abs(attr))
            
            print(f"Processed batch {batch_idx + 1}/{n_batches}")

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    return aggregated_importance, results_dir


def visualize_and_save_results(aggregated_importance, results_dir):
    """
    Visualize and save feature importance results.

    Args:
        aggregated_importance (dict): Dictionary of feature importances.
        results_dir (str): Directory to save visualization.
    """
    mean_importance = {feat: np.mean(values) for feat, values in aggregated_importance.items()}
    total = sum(mean_importance.values())
    feature_importance = {k: v / total for k, v in mean_importance.items()}

    # Sort and visualize top 20 features
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    top_features = dict(list(sorted_importance.items())[:20])

    plt.figure(figsize=(15, 10))
    plt.barh(list(top_features.keys()), list(top_features.values()))
    plt.title("Integrated Gradients Feature Importance (Top 20)")
    plt.xlabel("Mean Attribution Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()


def main():
    """Main function to calculate and visualize Integrated Gradients feature importance."""
    parser = argparse.ArgumentParser(description="Calculate Integrated Gradients feature importance")
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--saved_data_dir', required=True, help='Directory containing saved data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=44, help='Random seed for reproducibility')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.load(os.path.join(args.saved_data_dir, "ig_test_data.pt"))
    x_test, mask_test, config = test_data['x_test'], test_data['mask_test'], test_data['config']
    model = load_trained_model(args.model_path, config).to(device)

    aggregated_importance, results_dir = calculate_ig_importance(model, x_test, mask_test, config, device, args.batch_size, args.seed)
    visualize_and_save_results(aggregated_importance, results_dir)

    print(f"Results saved in {results_dir}")


if __name__ == "__main__":
    main()
