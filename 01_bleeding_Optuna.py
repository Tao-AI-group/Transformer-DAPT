import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pycox.models import PCHazard
from pycox.evaluation import EvalSurv
import json
from easydict import EasyDict

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Local imports
from modules.model import Transformer_DAPT
from modules.train_utils import Trainer
from modules.config import STConfig

import optuna
from optuna.pruners import MedianPruner


# Set CUDA devices 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import logging
import datetime


def setup_logging(script_dir):
    """
    Configure and initialize logging for the script.
    
    This function creates a 'logs' directory within the given script directory (if it does not
    already exist) and sets up both file-based and console logging. Log messages will be 
    formatted to include timestamps, log levels, and the message itself.
    
    Args:
        script_dir (str): The directory path where the script is located.

    Returns:
        str: The file path to the newly created log file.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'optuna_run_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()         # Also output logs to console
        ]
    )
    return log_file


class FeatureMapper:
    """
    A utility class to map between feature indices and their names.
    
    For multi-hot encoded features, this class helps to reconstruct the 
    original feature names from model inputs. This is especially helpful 
    for model interpretability, allowing you to identify which specific 
    features are active for a given sample.
    """
    def __init__(self, feature_names):
        """
        Initialize the FeatureMapper with a list of feature names.

        Args:
            feature_names (list or array-like of str): 
                The list of all feature column names.
        """
        self.feature_names = list(feature_names)
        # Create two dictionaries: index->feature_name and feature_name->index
        self.index_to_feature = {i: name for i, name in enumerate(feature_names)}
        self.feature_to_index = {name: i for i, name in enumerate(feature_names)}
        # Assign a padding index (used to pad sequences with no additional meaning)
        self.padding_idx = len(feature_names)
    
    def get_feature_names(self, indices):
        """
        Convert a sequence of feature indices (excluding padding) back into human-readable feature names.
        
        Args:
            indices (list of int): The indices to be converted into feature names.

        Returns:
            list of str: The corresponding feature names for the given indices.
        """
        return [self.index_to_feature[idx] for idx in indices if idx != self.padding_idx]
    
    def get_sample_features(self, model_input, attention_mask):
        """
        For a given sample, retrieve the active (non-padded) features 
        based on the attention mask.
        
        Args:
            model_input (torch.Tensor): 1D tensor of feature indices for a single sample.
            attention_mask (torch.Tensor): 1D tensor indicating which positions are valid (1) vs. padded (0).

        Returns:
            list of str: The names of the active features for this sample.
        """
        active_indices = model_input[attention_mask.bool()].cpu().numpy()
        return self.get_feature_names(active_indices)


def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries: Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target(df):
    """
    Extract the target variables (duration and event status) from a DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing 'duration' and 'event' columns.

    Returns:
        tuple of np.ndarray: (duration array, event array).
    """
    return df['duration'].values, df['event'].values


def analyze_feature_distribution(df, cols_categorical):
    """
    Analyze and print statistics about the distribution of active features per sample.

    This helps decide the maximum sequence length (max_active_features) for the model.
    It prints summary statistics like min, max, mean, median, and certain percentiles 
    of the number of active features.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cols_categorical (list of str): The column names corresponding to multi-hot features.

    Returns:
        int: The maximum number of active features observed in any sample.
    """
    # Each sample’s total active features is simply the sum of its binary feature columns.
    active_counts = df[cols_categorical].sum(axis=1)
    print("\nFeature Distribution Statistics:")
    print(f"Min active features: {active_counts.min()}")
    print(f"Max active features: {active_counts.max()}")
    print(f"Mean active features: {active_counts.mean():.2f}")
    print(f"Median active features: {active_counts.median():.2f}")
    print(f"90th percentile: {np.percentile(active_counts, 90):.2f}")
    print(f"95th percentile: {np.percentile(active_counts, 95):.2f}")

    # Use the maximum number of features to set max_active_features
    max_features = int(active_counts.max())
    print(f"\nUsing maximum number of features: {max_features}")
    return max_features


def get_active_features(data, cols_categorical, max_features):
    """
    Convert binary feature columns to lists of active feature indices.

    If the number of active features exceeds max_features, it truncates
    by randomly selecting a subset. Also keeps track of samples that were 
    truncated for potential debugging.

    Args:
        data (pd.DataFrame): The dataset subset (e.g., train, val, or test).
        cols_categorical (list of str): The column names for multi-hot features.
        max_features (int): The maximum allowed features per sample.

    Returns:
        tuple:
            feature_lists (list of list of int): 
                Each sub-list contains the indices of active features for that sample.
            feature_names_lists (list of list of str):
                Parallel list, each sub-list contains the names of active features for that sample.
    """
    feature_lists = []
    feature_names_lists = []
    truncation_count = 0
    truncated_samples = []
    
    for idx, row in data[cols_categorical].iterrows():
        # Identify which features are active (==1)
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


def create_study():
    """
    Create an Optuna study object for hyperparameter optimization.

    This function configures the study with:
      - A direction of "maximize" (for c-index).
      - A MedianPruner to prune poor-performing trials.
      - A TPE sampler with a fixed seed.

    Returns:
        optuna.study.Study: A configured Optuna study instance.
    """
    return optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            seed=44,
            multivariate=True  # Use multivariate TPE
        )
    )


def objective(trial, study, data_tuple, base_config):
    """
    Objective function for Optuna hyperparameter optimization.

    At a high level, this function:
      1. Copies the base configuration and updates it with hyperparameters sampled by Optuna.
      2. Transforms the survival labels for the training/validation data.
      3. Creates a new model and Trainer instance using these hyperparameters.
      4. Trains the model and evaluates on the validation set, returning the c-index as the optimization metric.
      5. If the performance is improved, saves the best model checkpoint.

    Args:
        trial (optuna.trial.Trial): The current trial object.
        study (optuna.study.Study): The study to which this trial belongs.
        data_tuple (tuple): A pre-processed data tuple containing:
            (x_train, mask_train), (x_val, mask_val), (x_test, mask_test),
             df_y_train, df_y_val, df_y_test, df_train, df_val, df_test, config.
        base_config (EasyDict): A base configuration object with default settings.

    Returns:
        float: The validation c-index, which Optuna will use to rank hyperparameter configurations.
    """
    set_seed(44)

    # Unpack data
    (x_train, mask_train), (x_val, mask_val), (x_test, mask_test), \
    df_y_train, df_y_val, df_y_test, df_train, df_val, df_test, config = data_tuple
    
    # Copy base config so each trial starts fresh
    config = EasyDict(base_config.copy())
    
    # Suggest hyperparameters
    config.lr = trial.suggest_float('lr', 2e-4, 5e-3, log=True)
    config.batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    config.embedding_size = trial.suggest_categorical('embedding_size', [64, 128])
    config.num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 3)
    config.weight_decay = trial.suggest_float('weight_decay', 1e-2, 2e-2, log=True)
    config.Triplet_weight = trial.suggest_float('Triplet_weight', 0.4, 0.8)
    config.BCE_weight = trial.suggest_float('BCE_weight', 0.3, 0.5)
    config.num_durations = trial.suggest_int('num_durations', 21, 24)
    config.Hazard_weight = trial.suggest_float('Hazard_weight', 0.75, 0.95)
    config.num_attention_heads = trial.suggest_categorical('num_attention_heads', [2, 4, 8])
    
    # Set dependent parameters based on above
    config.hidden_size = config.embedding_size
    config.intermediate_size = config.hidden_size

    # Prepare a PyCox label transform with the updated number of durations
    labtrans = PCHazard.label_transform(config.num_durations)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    # Update the y DataFrames with new transformations
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

    # Update config with new label transform info
    config.labtrans = labtrans
    config.duration_index = labtrans.cuts
    config.out_feature = labtrans.out_features
    
    # Log hyperparameters
    logging.info("Trial hyperparameters:")
    for param_name, param_value in trial.params.items():
        logging.info(f"{param_name}: {param_value}")
    logging.info(f"\nTrial {trial.number} - Starting training with parameters: {trial.params}")
    
    # Initialize and train the model
    torch.manual_seed(44)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(44)
    model = Transformer_DAPT(config)
    trainer = Trainer(model)
    train_loss, val_loss = trainer.fit(
        train_set=((x_train, mask_train), df_y_train),
        val_set=((x_val, mask_val), df_y_val),
        batch_size=config['batch_size'],
        val_batch_size=config['batch_size'],
        epochs=200,
        learning_rate=config['lr'],
        weight_decay=config['weight_decay'],
        optimizer=config['optimizer_name']
    )

    # Evaluate on validation set using c-index
    model.sub = config['num_sub']  # Additional config usage if needed
    val_surv = model.predict_surv_df((x_val, mask_val))
    durations_val = df_val['duration'].values
    events_val = df_val['event'].values
    ev_val = EvalSurv(val_surv, durations_val, events_val, censor_surv='km')
    val_c_index = ev_val.concordance_td(config['c_td_version'])
    logging.info(f"Trial {trial.number} - Validation c-index: {val_c_index:.4f}")
    
    # Evaluate test c-index for logging (not used by Optuna for ranking)
    test_surv = model.predict_surv_df((x_test, mask_test))
    durations_test = df_test['duration'].values
    events_test = df_test['event'].values
    ev_test = EvalSurv(test_surv, durations_test, events_test, censor_surv='km')
    test_c_index = ev_test.concordance_td(config['c_td_version'])
    logging.info(f"Trial {trial.number} - Test c-index: {test_c_index:.4f}")

    # Save the best model checkpoint if current trial is the best so far
    script_dir = os.path.dirname(os.path.abspath(__file__))
    best_results_dir = os.path.join(script_dir, 'best_results')
    os.makedirs(best_results_dir, exist_ok=True)

    # Check if this trial is better than the best so far (or if it's the first)
    if study.trials[-1].number == 0 or val_c_index > study.best_value:
        config_dict = dict(config)  # Convert EasyDict to a normal dict
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config_dict,
            'val_c_index': val_c_index,
            'test_c_index': test_c_index,
            'trial_params': trial.params
        }
        checkpoint_path = os.path.join(best_results_dir, 'best_model.pt')
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved best model to {checkpoint_path} with val_c_index: {val_c_index:.4f}")
    
    return val_c_index


def load_data(config, df):
    """
    Load and preprocess data for survival analysis.

    The typical workflow here is:
      1. Identify which columns are multi-hot features and exclude duration/event columns.
      2. Create a FeatureMapper for interpretability.
      3. Analyze and determine the maximum active features for any sample.
      4. Split the dataset into train/validation/test sets.
      5. (Optional) Upsample the minority class (event==1) if config.balanced is True.
      6. Convert the multi-hot representation into lists of active feature indices.
      7. Pad sequences to a fixed length and create attention masks.
      8. Transform survival labels using PyCox label transform.
    
    Args:
        config (EasyDict): A configuration object with various parameters.
        df (pd.DataFrame): The full dataset, including binary feature columns, 
            'duration', and 'event'.

    Returns:
        tuple: 
            (x_train, attention_mask_train), (x_val, attention_mask_val),
            (x_test, attention_mask_test), df_y_train, df_y_val, df_y_test,
            df_train, df_val, df_test, config
    """
    # Identify feature columns (exclude duration and event)
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
        Pad sequences to a fixed length with a padding_idx.

        Args:
            sequences (list of list of int): 
                Each sub-list is a list of active feature indices for a single sample.
            max_len (int): The length to which each sequence should be padded.
            padding_idx (int): The index used for padding.

        Returns:
            torch.Tensor: A 2D tensor of shape (num_samples, max_len) containing padded sequences.
        """
        padded = []
        for seq in sequences:
            padded_seq = seq + [padding_idx] * (max_len - len(seq))
            padded.append(padded_seq)
        return torch.tensor(padded, dtype=torch.long)

    # Split data into train+val and test sets
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

    # Optionally, rebalance training data if config.balanced is True
    if config.balanced:
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

    # Pad sequences
    x_train = pad_sequences(x_train_indices, config.max_active_features)
    x_val = pad_sequences(x_val_indices, config.max_active_features)
    x_test = pad_sequences(x_test_indices, config.max_active_features)

    # Create attention masks (1 for active features, 0 for padding)
    attention_mask_train = (x_train != num_features).float()
    attention_mask_val = (x_val != num_features).float()
    attention_mask_test = (x_test != num_features).float()

    # Label transformation for survival modeling (PyCox)
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
    
    df_y_test = df_test[['duration', 'event']]  # Test set remains with original labels

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


def interpret_predictions(model, x_test, mask_test, feature_mapper, n_samples=5):
    """
    Print out the active features for a few test samples to understand what the model sees.

    Args:
        model (nn.Module): The trained model (not explicitly needed for these prints, 
            but could be used for further interpretability).
        x_test (torch.Tensor): Padded feature indices for test samples.
        mask_test (torch.Tensor): Corresponding attention masks (1=active, 0=padding).
        feature_mapper (FeatureMapper): The mapper to convert indices to feature names.
        n_samples (int): Number of samples to interpret from the test set.
    """
    for i in range(n_samples):
        x = x_test[i].unsqueeze(0)
        mask = mask_test[i].unsqueeze(0)
        
        # Retrieve active features for sample i
        active_features = feature_mapper.get_sample_features(x[0], mask[0])
        print(f"\nSample {i+1} Active Features ({len(active_features)}):")
        print(active_features)


def save_config(config, save_dir):
    """
    Save the model configuration object to disk using torch.save.

    Args:
        config (EasyDict or dict): The configuration object to save.
        save_dir (str): The directory where the configuration should be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_save_path = os.path.join(save_dir, "model_config.pt")
    torch.save(config, config_save_path)
    print(f"Config saved to {config_save_path}")


def main():
    """
    The main entry point for data loading, training, hyperparameter optimization, 
    and final evaluation of a multi-head attention-based survival model.

    Workflow:
      1. Set random seeds for reproducibility.
      2. Initialize logging and specify the data file path.
      3. Load the CSV dataset, rename columns, and reorder columns for convenience.
      4. Create a base configuration using STConfig.
      5. Preprocess the data (split train/val/test, create feature mappings, etc.).
      6. Create an Optuna study and optimize hyperparameters.
      7. After finding the best trial, load the best model and evaluate it thoroughly 
         on both validation and test sets, computing survival metrics (c-index, IBS) 
         as well as standard classification metrics (accuracy, F1, AUROC, AUPRC).
      8. Log metrics to console and save them to a JSON file.

    This script can run up to the specified number of trials. 
    It also supports early pruning of poor trials via MedianPruner.
    """
    set_seed(44)

    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Setup logging
    log_file = setup_logging(script_dir)
    logging.info("Starting Optuna optimization")

    # Log file paths and basic info
    file_path = os.path.join(
        script_dir, "../../Results/Results_v1/Results_0d_365d_window/bleeding_multihot_encoded.csv"
    )
    logging.info(f"Data file path: {file_path}")

    # Load the initial DataFrame
    df = pd.read_csv(file_path)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    # Drop columns containing 'set' to avoid confusion, rename "tte" -> "duration" and "Label" -> "event"
    df = df[[col for col in df.columns if "set" not in col]]
    df = df.rename(columns={"tte": "duration", "Label": "event"})
    df = df.drop(columns=["Pt_id"])
    
    # Reorder columns so that 'event' and 'duration' are at the end
    df = pd.concat([df.drop(columns=["event", "duration"]), df[["event", "duration"]]], axis=1)

    # Initialize base configuration
    base_config = STConfig
    # Calculate class_weight as ratio of positives to negatives (optional usage)
    base_config.class_weight = df['event'].value_counts(normalize=True).get(1, 0) / \
                              df['event'].value_counts(normalize=True).get(0, 1)
    base_config.optimizer_name = "AdamW"
    base_config.balanced = True
    base_config.focal = False
    base_config.num_sub = 1
    base_config.c_td_version = "antolini"
    base_config.early_stop_patience = 15
    base_config.seed = 44

    # Load and preprocess data once
    data_tuple = load_data(base_config, df)

    # Create and run study
    study = create_study()
    logging.info("Created Optuna study")

    study.optimize(
        lambda trial: objective(trial, study, data_tuple, base_config),
        n_trials=2,            # Number of trials for demonstration (increase as needed)
        gc_after_trial=True,   # Garbage collection after each trial
        show_progress_bar=True
    )

    # Log best results
    logging.info("\nOptimization finished")
    logging.info(f"Best trial parameters: {study.best_trial.params}")
    logging.info(f"Best validation C-index: {study.best_trial.value:.4f}")
    
    # Load the best model checkpoint
    checkpoint_path = os.path.join(script_dir, 'best_results', 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    saved_config_dict = checkpoint['config']
    saved_config = EasyDict(saved_config_dict)

    # Create the model and load its state
    model = Transformer_DAPT(saved_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Unpack data again for final evaluation
    (x_train, mask_train), (x_val, mask_val), (x_test, mask_test), \
    df_y_train, df_y_val, df_y_test, df_train, df_val, df_test, config = data_tuple

    model.eval()  # Switch to evaluation mode
    model.sub = saved_config.num_sub

    logging.info("\n=== Final Model Evaluation ===")
    print("\n=== Final Model Evaluation ===")

    # Validation metrics
    logging.info("\nCalculating validation metrics...")
    print("\nCalculating validation metrics...")
    val_surv = model.predict_surv_df((x_val, mask_val))
    durations_val = df_val['duration'].values
    events_val = df_val['event'].values
    ev_val = EvalSurv(val_surv, durations_val, events_val, censor_surv='km')
    val_c_index = ev_val.concordance_td(config.c_td_version)
    val_ibs = ev_val.integrated_brier_score(model.duration_index)
    logging.info(f"Validation Concordance Index (C_index): {val_c_index:.4f}")
    logging.info(f"Validation Integrated Brier Score: {val_ibs:.4f}")
    print(f"Validation Concordance Index (C_index): {val_c_index:.4f}")
    print(f"Validation Integrated Brier Score: {val_ibs:.4f}")

    # Test metrics
    logging.info("\nCalculating test metrics...")
    print("\nCalculating test metrics...")
    surv = model.predict_surv_df((x_test, mask_test))
    durations_test = df_test['duration'].values
    events_test = df_test['event'].values
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td(config.c_td_version)
    test_ibs = ev.integrated_brier_score(model.duration_index)
    logging.info(f"Test Concordance Index (C_index): {c_index:.4f}")
    logging.info(f"Test Integrated Brier Score: {test_ibs:.4f}")
    print(f"Test Concordance Index (C_index): {c_index:.4f}")
    print(f"Test Integrated Brier Score: {test_ibs:.4f}")

    # Standard classification metrics
    logging.info("\nCalculating standard classification metrics...")
    print("\nCalculating standard classification metrics...")
    binary_score = model.predict_binary((x_test, mask_test))  # Probability of event
    predictions_standard = (binary_score > 0.5).numpy()
    
    accuracy = accuracy_score(events_test, predictions_standard)
    f1 = f1_score(events_test, predictions_standard)
    auroc = roc_auc_score(events_test, binary_score)
    auprc = average_precision_score(events_test, binary_score)

    logging.info("\nStandard Metrics (threshold = 0.5):")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 score: {f1:.4f}")
    logging.info(f"AUROC: {auroc:.4f}")
    logging.info(f"AUPRC: {auprc:.4f}")
    
    print("\nStandard Metrics (threshold = 0.5):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

    # Save final metrics
    metrics_dict = {
        'validation_c_index': val_c_index,
        'validation_integrated_brier_score': val_ibs,
        'test_c_index': c_index,
        'test_integrated_brier_score': test_ibs,
        'test_accuracy': accuracy,
        'test_f1': f1,
        'test_auroc': auroc,
        'test_auprc': auprc
    }

    metrics_file = os.path.join(script_dir, 'best_results', 'final_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"\nSaved all metrics to: {metrics_file}")

if __name__ == "__main__":
    main()
