import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pycox.models import PCHazard
from pycox.evaluation import EvalSurv
import pickle
from easydict import EasyDict

# Local imports - these are custom modules for the survival model implementation
from modules.model import Transformer_DAPT
from modules.config import STConfig

# Set CUDA device for GPU acceleration
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class FeatureMapper:
    """Maps between feature indices and their names to improve model interpretability."""
    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.index_to_feature = {i: name for i, name in enumerate(feature_names)}
        self.feature_to_index = {name: i for i, name in enumerate(feature_names)}
        self.padding_idx = len(feature_names)
    
    def get_feature_names(self, indices):
        """Convert feature indices (excluding padding) back to feature names."""
        return [self.index_to_feature[idx] for idx in indices if idx != self.padding_idx]

def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target(df):
    """
    Extract the target variables (duration and event status) from the DataFrame.
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


def calculate_time_dependent_metrics(surv_df, times, durations, events):
    """
    Calculate time-dependent concordance indices at specific time points.
    
    Args:
        surv_df: Survival predictions DataFrame from model.predict_surv_df()
        times: List of time points to evaluate (e.g., [30, 60, 90] days)
        durations: Array of observed durations
        events: Array of event indicators (0=censored, 1=event occurred)
    
    Returns:
        Dictionary of time-specific concordance indices
    """
    time_specific_metrics = {}
    
    # Validate inputs
    if not isinstance(surv_df, pd.DataFrame):
        raise TypeError("surv_df must be a pandas DataFrame")
    
    for t in times:
        # Create censored data for this time point
        censored_durations = np.minimum(durations, t)
        censored_events = events * (durations <= t)
        
        # Create EvalSurv object with censored data
        ev_t = EvalSurv(surv_df, censored_durations, censored_events, censor_surv='km')
        
        # Calculate concordance index
        c_index_t = ev_t.concordance_td('antolini')
        time_specific_metrics[f'{t}_days'] = c_index_t
        
    return time_specific_metrics


def calculate_p_value(bootstrap_values, null_value=0.5):
    """
    Calculate two-sided p-value from bootstrap distribution.
    
    Args:
        bootstrap_values: Array of bootstrap estimates
        null_value: Null hypothesis value (default: 0.5 for C-index)
    
    Returns:
        Two-sided p-value
    """
    # Calculate test statistic (mean of bootstrap distribution)
    observed = np.mean(bootstrap_values)
    
    # Calculate two-sided p-value
    if observed >= null_value:
        p_value = 2 * (1 - np.mean(bootstrap_values >= null_value))
    else:
        p_value = 2 * np.mean(bootstrap_values <= null_value)
    
    return p_value


def bootstrap_metric(surv_df, times, durations, events, n_bootstrap=1000, alpha=0.95):
    """
    Calculate time-dependent metrics with bootstrap confidence intervals and p-values.
    
    Args:
        surv_df: Survival predictions DataFrame
        times: List of time points to evaluate
        durations: Array of observed durations
        events: Array of event indicators
        n_bootstrap: Number of bootstrap samples
        alpha: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Dictionary containing mean metrics, confidence intervals, and p-values
    """
    n_samples = len(durations)
    bootstrap_metrics = {f'{t}_days': [] for t in times}
    
    print(f"\nPerforming {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_bootstrap} iterations...")
            
        # Generate bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Get bootstrap sample
        bootstrap_durations = durations[indices]
        bootstrap_events = events[indices]
        bootstrap_surv = surv_df.iloc[:, indices]
        
        # Calculate metrics for this bootstrap sample
        metrics = calculate_time_dependent_metrics(bootstrap_surv, times, bootstrap_durations, bootstrap_events)
        
        # Store results
        for t in times:
            bootstrap_metrics[f'{t}_days'].append(metrics[f'{t}_days'])
    
    # Calculate statistics
    results = {}
    for t in times:
        key = f'{t}_days'
        values = np.array(bootstrap_metrics[key])
        
        # Calculate mean and confidence intervals
        mean_value = np.mean(values)
        lower = np.percentile(values, (1 - alpha) * 100 / 2)
        upper = np.percentile(values, 100 - (1 - alpha) * 100 / 2)
        
        # Calculate p-value
        p_value = calculate_p_value(values)
        
        results[key] = {
            'mean': mean_value,
            'ci_lower': lower,
            'ci_upper': upper,
            'p_value': p_value
        }
    
    return results

def print_metrics_with_ci_and_pvalue(metrics_dict):
    """
    Pretty print metrics with confidence intervals and p-values
    """
    print("\nTime-specific Concordance Indices with 95% Confidence Intervals and P-values:")
    time_mapping = {
        '30_days': '1 Month (30 days)',
        '60_days': '2 Months (60 days)',
        '90_days': '3 Months (90 days)',
        '180_days': '6 Months (180 days)',
        '270_days': '9 Months (270 days)',
        '365_days': '12 Months (365 days)'
    }
    
    for key, label in time_mapping.items():
        if key in metrics_dict:
            metric = metrics_dict[key]
            p_value_str = f"p < 0.001" if metric['p_value'] < 0.001 else f"p = {metric['p_value']:.3f}"
            print(f"{label}: {metric['mean']:.2f} (95% CI: {metric['ci_lower']:.2f} - {metric['ci_upper']:.2f}), {p_value_str}")


def get_active_features(data, cols_categorical, max_features):
    """
    Convert binary feature columns to lists of active feature indices.
    If a sample exceeds max_features, truncate by randomly selecting a subset.
    
    Args:
        data: DataFrame containing binary features
        cols_categorical: List of column names for categorical/binary features
        max_features: Maximum number of features to keep per sample
    
    Returns:
        feature_lists: List of active feature indices for each sample
        feature_names_lists: List of active feature names for each sample
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
    Load and preprocess data for the survival analysis model:

    
    Args:
        config: Configuration object with model parameters
        df: DataFrame containing features, duration, and event indicators
        
    Returns:
        Processed data ready for model training and evaluation
    """
    # Identify feature columns
    exclude_columns = ['duration', 'event']
    cols_categorical = [col for col in df.columns if col not in exclude_columns]
    num_features = len(cols_categorical)
    
    # Create a feature mapper for interpretability
    feature_mapper = FeatureMapper(cols_categorical)
    config.feature_mapper = feature_mapper
    
    # Determine the maximum sequence length based on feature distribution
    config.max_active_features = analyze_feature_distribution(df, cols_categorical)
    
    def pad_sequences(sequences, max_len, padding_idx=num_features):
        """Pad sequences to a fixed length with padding_idx."""
        padded = []
        for seq in sequences:
            padded_seq = seq + [padding_idx] * (max_len - len(seq))
            padded.append(padded_seq)
        return torch.tensor(padded, dtype=torch.long)

    # Split data into train+val and test sets (stratify by event status)
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

    print("\nData Split: Train={}, Val={}, Test={}".format(len(df_train), len(df_val), len(df_test)))
    
    # Store original training indices for reference
    config.original_train_idx = train_idx_final

    # Optional: Rebalance training data if config.balanced is True
    if config.balanced:
        # Up-sample minority class (event=1) to match majority class (event=0)
        df_majority = df_train[df_train['event'] == 0]
        df_minority = df_train[df_train['event'] == 1]
        minority_upsampled_idx = np.random.choice(
            df_minority.index, size=len(df_majority), replace=True
        )
        df_minority_upsampled = df.loc[minority_upsampled_idx]
        df_train = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=config.seed)
        print(f"Balanced training data - New size: {len(df_train)}, New event rate: {df_train['event'].mean():.3f}")

    # Convert binary features into active indices
    x_test_indices, x_test_features = get_active_features(df_test, cols_categorical, config.max_active_features)

    # We only need test data for model comparison
    x_train_indices, x_train_features = get_active_features(df_train, cols_categorical, config.max_active_features)
    x_val_indices, x_val_features = get_active_features(df_val, cols_categorical, config.max_active_features)

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
    
    # We need to fit_transform on training data to properly setup labtrans
    y_train = labtrans.fit_transform(*get_target(df_train))

    print("\nLabel Transform Info:")
    print(f"Number of duration cuts: {len(labtrans.cuts)}")
    print(f"Duration cuts: {labtrans.cuts}")
    print(f"Output features: {labtrans.out_features}")

    # Update config with label transformation info
    config.labtrans = labtrans
    config.num_categorical_feature = num_features + 1  # +1 for padding token
    config.padding_idx = num_features
    config.duration_index = labtrans.cuts
    config.out_feature = labtrans.out_features
    config.num_feature = config.max_active_features

    return (x_test, attention_mask_test), df_test, config

def main():
    """
    Main function to generate TDAPT_results.pkl for model comparison.
    Loads the pre-trained model and saves predictions on test data.
    """
    # Set random seed for reproducibility
    set_seed(44)
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        script_dir, "../../Results/Results_v1/Results_0d_365d_window/bleeding_multihot_encoded.csv"
    )

    # Load the initial DataFrame
    df = pd.read_csv(file_path)
    df = df[[col for col in df.columns if "set" not in col]]
    df = df.rename(columns={"tte": "duration", "Label": "event"})
    df = df.drop(columns=["Pt_id"])
    
    # Ensure the event and duration columns are at the end
    df = pd.concat([df.drop(columns=["event", "duration"]), df[["event", "duration"]]], axis=1)

    # Initialize configuration
    config = STConfig
    config.class_weight = df['event'].value_counts(normalize=True).get(1, 0) / \
                          df['event'].value_counts(normalize=True).get(0, 1)
    
    # Model configuration
    config.balanced = True
    config.num_durations = 23
    config.num_sub = 1
    config.c_td_version = "antolini"
    config.seed = 44
    
    # Load and preprocess data (focus only on test data for prediction)
    (x_test, mask_test), df_test, config = load_data(config, df)


    # Load the best pre-trained model
    checkpoint_path = os.path.join(script_dir, 'final_model', 'best_model.pt')
    print(f"\nLoading pre-trained model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    saved_config_dict = checkpoint['config']
    
    # Convert dict back to EasyDict
    saved_config = EasyDict(saved_config_dict)

    # Create model with the saved configuration
    model = Transformer_DAPT(saved_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Set model parameters from saved config
    model.sub = saved_config.num_sub

    # Generate predictions on test data
    print("\nGenerating predictions on test data...")
    surv = model.predict_surv_df((x_test, mask_test))
    durations_test = df_test['duration'].values
    events_test = df_test['event'].values

    # Save results for model comparison
    results = {
        'predictions': surv,
        'durations': durations_test,
        'events': events_test
    }
    
    # Create directory for results
    results_dir = os.path.join(script_dir, 'model_comparison_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions to pickle file
    results_path = os.path.join(results_dir, 'TDAPT_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {results_path}")
    
    # Calculate overall metrics
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td(saved_config.c_td_version)

    print(f"\nOverall Performance Metrics:")
    print(f"Test Concordance Index (C_index): {c_index:.2f}")

    # Calculate time-specific metrics with bootstrapping
    print("\nCalculating time-specific metrics with bootstrapping...")
    evaluation_times = [30, 60, 90, 180, 270, 365]  # Time points in days
    try:
        # Calculate point estimates
        time_specific_metrics = calculate_time_dependent_metrics(surv, evaluation_times, durations_test, events_test)
        
        print("\nPoint Estimates:")
        print(f"1 Month (30 days): {time_specific_metrics['30_days']:.2f}")
        print(f"2 Months (60 days): {time_specific_metrics['60_days']:.2f}")
        print(f"3 Months (90 days): {time_specific_metrics['90_days']:.2f}")
        print(f"6 Months (180 days): {time_specific_metrics['180_days']:.2f}")
        print(f"9 Months (270 days): {time_specific_metrics['270_days']:.2f}")
        print(f"12 Months (365 days): {time_specific_metrics['365_days']:.2f}")
        
        # Calculate bootstrapped estimates with confidence intervals and p-values
        bootstrapped_metrics = bootstrap_metric(surv, evaluation_times, durations_test, events_test, n_bootstrap=1000)
        print_metrics_with_ci_and_pvalue(bootstrapped_metrics)
        
    except Exception as e:
        print(f"Error calculating time-specific metrics: {str(e)}")

    # Standard classification metrics
    print("\nCalculating standard classification metrics...")
    binary_score = model.predict_binary((x_test, mask_test))
    predictions_standard = (binary_score > 0.5).numpy()
    
if __name__ == "__main__":
    main()