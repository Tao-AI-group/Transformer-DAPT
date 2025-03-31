import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pycox.models import PCHazard
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt
import json
from easydict import EasyDict
from datetime import datetime

# Local imports
from modules.model import Transformer_DAPT
from modules.config import STConfig
from modules.calibration_advanced_V2 import CalibratedModelEvaluator

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class FeatureMapper:
    """
    A utility class to map between feature indices and their names. 
    This is helpful for model interpretability, as it allows users to identify 
    which features are active for a given sample.
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
            indices: List of feature indices
            
        Returns:
            list: Feature names corresponding to the provided indices
        """
        return [self.index_to_feature[idx] for idx in indices if idx != self.padding_idx]
    
    def get_sample_features(self, model_input, attention_mask):
        """
        For a given sample, retrieve the active features (non-padded ones)
        based on the attention mask.
        
        Args:
            model_input: Tensor containing feature indices
            attention_mask: Tensor indicating which positions are active (1) vs padding (0)
            
        Returns:
            list: Names of active features in the sample
        """
        active_indices = model_input[attention_mask.bool()].cpu().numpy()
        return self.get_feature_names(active_indices)


def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target(df):
    """
    Extract the target variables (duration and event status) from the DataFrame.
    
    Args:
        df: DataFrame containing duration and event columns
        
    Returns:
        tuple: Arrays of duration values and event indicators
    """
    return df['duration'].values, df['event'].values


def analyze_feature_distribution(df, cols_categorical):
    """Analyze distribution of active features per sample."""
    active_counts = df[cols_categorical].sum(axis=1)
    print(f"\nFeature Distribution: Min={active_counts.min()}, Max={active_counts.max()}, Mean={active_counts.mean():.2f}")
    
    # Use the maximum number of features to set max_active_features
    max_features = int(active_counts.max())
    print(f"Using maximum number of features: {max_features}")
    return max_features

def get_active_features(data, cols_categorical, max_features):
    """
    Convert binary feature columns to lists of active feature indices.
    If a sample exceeds max_features, truncate by randomly selecting a subset.
    
    Args:
        data: DataFrame with binary feature columns
        cols_categorical: List of categorical feature column names
        max_features: Maximum number of features to keep per sample
        
    Returns:
        tuple: Lists of active feature indices and their corresponding names
    """
    feature_lists = []
    feature_names_lists = []
    truncation_count = 0
    truncated_samples = []
    
    for idx, row in data[cols_categorical].iterrows():
        active_indices = np.where(row == 1)[0]
        active_features = np.array(cols_categorical)[active_indices]
        
        # If more than max_features are present, randomly select a subset
        if len(active_indices) > max_features:
            truncation_count += 1
            truncated_samples.append({
                'original_count': len(active_indices),
                'sample_id': idx
            })
            selected_idx = random.sample(range(len(active_indices)), max_features)
            active_indices = active_indices[selected_idx]
            active_features = active_features[selected_idx]
            
        feature_lists.append(active_indices.tolist())
        feature_names_lists.append(active_features.tolist())
    
    if truncation_count > 0:
        print(f"\nTruncated {truncation_count} samples ({truncation_count/len(data)*100:.2f}% of data)")
        print("Samples requiring truncation:", truncated_samples[:5], "...")
    
    return feature_lists, feature_names_lists


def load_data(config, df):
    """
    Load and preprocess data:
    1. Split into training, validation, and test sets.
    2. Create a feature mapper for interpretability.
    3. Determine max sequence length (max_active_features).
    4. Convert binary features into active feature indices and pad sequences.
    5. Transform duration and event data for survival modeling with PyCox.
    
    Args:
        config: Configuration object
        df: DataFrame containing features and target variables
        
    Returns:
        tuple: Processed data and updated configuration
    """
    # Identify feature columns
    exclude_columns = ['duration', 'event']
    cols_categorical = [col for col in df.columns if col not in exclude_columns]
    num_features = len(cols_categorical)
    
    # Create a feature mapper for interpretability
    feature_mapper = FeatureMapper(cols_categorical)
    config.feature_mapper = feature_mapper
    
    # Determine the maximum sequence length
    config.max_active_features = analyze_feature_distribution(df, cols_categorical)
    
    def pad_sequences(sequences, max_len, padding_idx=num_features):
        """
        Pad sequences to a fixed length with padding_idx.
        
        Args:
            sequences: List of sequences to pad
            max_len: Maximum length to pad to
            padding_idx: Index to use for padding
            
        Returns:
            torch.Tensor: Padded sequences as a tensor
        """
        padded = []
        for seq in sequences:
            padded_seq = seq + [padding_idx] * (max_len - len(seq))
            padded.append(padded_seq)
        return torch.tensor(padded, dtype=torch.long)

    # Split data into train+val and test sets (stratified by event)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=config.seed, 
        stratify=df['event']
    )
    
    # Further split train+val into final train and validation sets
    df_train_val = df.loc[train_idx]
    df_test = df.loc[test_idx]
    train_idx_final, val_idx = train_test_split(
        df_train_val.index, test_size=1/8, random_state=config.seed,
        stratify=df_train_val['event']
    )
    
    df_val = df_train_val.loc[val_idx]
    df_train = df_train_val.loc[train_idx_final]

    # Print data statistics
    print("\nData Split Statistics:")
    print(f"Training set size: {len(df_train)}, Event rate: {df_train['event'].mean():.3f}")
    print(f"Validation set size: {len(df_val)}, Event rate: {df_val['event'].mean():.3f}")
    print(f"Test set size: {len(df_test)}, Event rate: {df_test['event'].mean():.3f}")

    print("\nDuration Statistics:")
    print("Training set - Mean duration: {:.2f}, Median: {:.2f}".format(
        df_train['duration'].mean(), df_train['duration'].median()))
    print("Validation set - Mean duration: {:.2f}, Median: {:.2f}".format(
        df_val['duration'].mean(), df_val['duration'].median()))
    print("Test set - Mean duration: {:.2f}, Median: {:.2f}".format(
        df_test['duration'].mean(), df_test['duration'].median()))
    
    # Store original training indices for reference
    config.original_train_idx = train_idx_final

    # Optional: Rebalance training data if config.balanced is True
    if config.balanced:
        # Upsample minority class (event=1) to match majority class (event=0)
        df_majority = df_train[df_train['event'] == 0]
        df_minority = df_train[df_train['event'] == 1]
        minority_upsampled_idx = np.random.choice(
            df_minority.index, size=len(df_majority), replace=True
        )
        df_minority_upsampled = df.loc[minority_upsampled_idx]
        df_train = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=config.seed)

    # Convert binary features into active indices
    x_train_indices, x_train_features = get_active_features(df_train, cols_categorical, config.max_active_features)
    x_val_indices, x_val_features = get_active_features(df_val, cols_categorical, config.max_active_features)
    x_test_indices, x_test_features = get_active_features(df_test, cols_categorical, config.max_active_features)

    # Store feature names for interpretability
    config.train_features = x_train_features
    config.val_features = x_val_features
    config.test_features = x_test_features

    # Pad sequences to fixed length
    x_train = pad_sequences(x_train_indices, config.max_active_features)
    x_val = pad_sequences(x_val_indices, config.max_active_features)
    x_test = pad_sequences(x_test_indices, config.max_active_features)

    # Create attention masks (1 for active features, 0 for padding)
    attention_mask_train = (x_train != num_features).float()
    attention_mask_val = (x_val != num_features).float()
    attention_mask_test = (x_test != num_features).float()

    # Label transformation for survival modeling with PCHazard
    labtrans = PCHazard.label_transform(config.num_durations)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    print("\nLabel Transform Debug:")
    print(f"Number of duration cuts: {len(labtrans.cuts)}")
    print(f"Duration cuts: {labtrans.cuts}")
    print(f"Out features: {labtrans.out_features}")

    # Create DataFrames for the transformed labels
    df_y_train = pd.DataFrame({
        "duration": y_train[0],
        "event": y_train[1],
        "fraction": y_train[2]
    }, index=df_train.index)
    
    df_y_val = pd.DataFrame({
        "duration": y_val[0],
        "event": y_val[1],
        "fraction": y_val[2]
    }, index=df_val.index)
    
    df_y_test = df_test[['duration', 'event']]  # test set remains with original labels

    # Update config with label transformation info
    config.labtrans = labtrans
    config.num_categorical_feature = num_features + 1  # +1 for padding token
    config.padding_idx = num_features
    config.duration_index = labtrans.cuts
    config.out_feature = labtrans.out_features
    config.num_feature = config.max_active_features

    return (x_train, attention_mask_train), (x_val, attention_mask_val), \
           (x_test, attention_mask_test), df_y_train, df_y_val, df_y_test, \
           df_train, df_val, df_test, config


def save_loaded_model_results(model, config, calibration_metrics, save_dir='calibration_model_results_V2'):
    """
    Save the model configuration and calibration metrics.
    
    Args:
        model: Trained model
        config: Configuration object
        calibration_metrics: Dictionary of calibration metrics
        save_dir: Directory to save results
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint with model, config, and metrics
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': dict(config),
        'calibration_metrics': calibration_metrics,
        'evaluation_timestamp': timestamp
    }
    
    # Save model checkpoint
    checkpoint_path = os.path.join(save_dir, f"loaded_model_evaluation_{timestamp}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Evaluation results saved to {checkpoint_path}")
    
    # Save metrics separately as JSON for easy access
    metrics_path = os.path.join(save_dir, f"calibration_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(calibration_metrics, f, indent=4)
    print(f"Calibration metrics saved to {metrics_path}")


def main():
    """
    Main function to run the survival model evaluation pipeline:
    1. Load and preprocess data
    2. Load pretrained model
    3. Evaluate model performance
    4. Perform calibration analysis
    5. Save results and plots
    """
    # Set seed for reproducibility
    set_seed(44)
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        script_dir, "../../Results/Results_v1/Results_0d_365d_window/bleeding_multihot_encoded.csv"
    )

    # Load the initial DataFrame
    df = pd.read_csv(file_path)
    df = df[[col for col in df.columns if "set" not in col]]  # Remove columns with "set" in name
    df = df.rename(columns={"tte": "duration", "Label": "event"})
    df = df.drop(columns=["Pt_id"])
    
    # Ensure the event and duration columns are at the end
    df = pd.concat([df.drop(columns=["event", "duration"]), df[["event", "duration"]]], axis=1)

    # Initialize configuration
    config = STConfig
    # Calculate class weight based on event distribution
    config.class_weight = df['event'].value_counts(normalize=True).get(1, 0) / \
                          df['event'].value_counts(normalize=True).get(0, 1)
    
    # Model configuration
    config.balanced = True  # Use class balancing
    config.num_durations = 23  # Number of time intervals for discrete-time survival model
    config.num_sub = 1  # Number of sub-networks
    config.c_td_version = "antolini"  # Concordance index version
    config.seed = 44  # Random seed

    # Load and preprocess data
    (x_train, mask_train), (x_val, mask_val), (x_test, mask_test), \
    df_y_train, df_y_val, df_y_test, df_train, df_val, df_test, config = load_data(config, df)

    # Load Best Model
    checkpoint_path = os.path.join(script_dir, 'final_model', 'best_model.pt')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    saved_config_dict = checkpoint['config']  # This is a dict

    # Convert dict back to EasyDict
    saved_config = EasyDict(saved_config_dict)

    # Create model with the saved configuration
    model = Transformer_DAPT(saved_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Use number of sub-networks from saved config
    model.sub = saved_config.num_sub

    # Validation metrics
    print("\nCalculating validation metrics...")
    val_surv = model.predict_surv_df((x_val, mask_val))
    durations_val = df_val['duration'].values
    events_val = df_val['event'].values
    ev_val = EvalSurv(val_surv, durations_val, events_val, censor_surv='km')
    val_c_index = ev_val.concordance_td(config.c_td_version)
    
    # Log and print validation metrics
    print(f"Validation Concordance Index (C_index): {val_c_index:3f}")

    # Test metrics
    print("\nCalculating test metrics...")
    surv = model.predict_surv_df((x_test, mask_test))
    durations_test = df_test['duration'].values
    events_test = df_test['event'].values
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    test_c_index = ev.concordance_td(config.c_td_version)
    
    print(f"Test Concordance Index (C_index): {test_c_index:.3f}")

    # Initialize calibration evaluator
    print("\nPerforming calibration analysis...")
    calibration_evaluator = CalibratedModelEvaluator(model)

    # Run calibration pipeline
    calibrated_model, calibration_metrics, _ = calibration_evaluator.run_calibration_pipeline(
        val_data=(x_val, mask_val),
        val_labels=df_y_val,
        test_data=(x_test, mask_test),
        test_labels=df_y_test,
        batch_size=saved_config.batch_size
    )

    # Get both uncalibrated and calibrated probabilities for test set
    model.eval()
    with torch.no_grad():
        test_uncal_probs = model.predict_binary((x_test, mask_test)).cpu().numpy()
        test_cal_probs = calibrated_model.predict_binary((x_test, mask_test)).cpu().numpy()
    test_labels = df_y_test['event'].values
    
    # Create enhanced calibration plot
    enhanced_fig = calibration_evaluator.create_enhanced_calibration_plot(
        test_uncal_probs,
        test_cal_probs,
        test_labels,
        n_bins=10
    )
    
    # Save plots
    plot_save_dir = os.path.join(script_dir, 'calibration_model_results_V2', 'plots')
    os.makedirs(plot_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    enhanced_fig.savefig(
        os.path.join(plot_save_dir, f"calibration_plot_enhanced_{timestamp}.png"),
        bbox_inches='tight',
        dpi=600
    )
    
    plt.close('all')
    
    # Save the model, config, and metrics
    save_loaded_model_results(model, saved_config, calibration_metrics)
    

if __name__ == "__main__":
    main()
