import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
import random
from pycox.models import DeepHitSingle
import torchtuples as tt
from torch import nn
import warnings
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set GPU device for computation
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def set_seed(seed):
    """
    Set random seed for reproducibility across all random number generators.
    This ensures experiments can be replicated with the same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CustomNet(nn.Module):
    """
    Custom neural network architecture for DeepHit model.
    
    A fully-connected neural network with batch normalization and dropout layers,
    designed specifically for survival analysis predictions.
    
    Args:
        in_features (int): Number of input features
        num_nodes (list): List containing the number of neurons in each hidden layer
        out_features (int): Number of output features (depends on the number of discrete time intervals)
    """
    def __init__(self, in_features, num_nodes, out_features):
        super().__init__()
        
        # Input layer normalization
        self.batch_norm1 = nn.BatchNorm1d(in_features)
        
        # First hidden layer
        self.lin1 = nn.Linear(in_features, num_nodes[0])
        self.batch_norm2 = nn.BatchNorm1d(num_nodes[0])
        self.dropout1 = nn.Dropout(0.4)
        
        # Second hidden layer
        self.lin2 = nn.Linear(num_nodes[0], num_nodes[1])
        self.batch_norm3 = nn.BatchNorm1d(num_nodes[1])
        self.dropout2 = nn.Dropout(0.4)
        
        # Third hidden layer
        self.lin3 = nn.Linear(num_nodes[1], num_nodes[2])
        self.batch_norm4 = nn.BatchNorm1d(num_nodes[2])
        self.dropout3 = nn.Dropout(0.4)
        
        # Output layer
        self.lin4 = nn.Linear(num_nodes[2], out_features)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Initialize weights using Kaiming initialization for better training dynamics
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, input):
        """Forward pass through the network"""
        # Input normalization
        x = self.batch_norm1(input)
        
        # First hidden layer
        x = self.lin1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.lin2(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Third hidden layer
        x = self.lin3(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Output layer (no activation, handled by DeepHitSingle model)
        x = self.lin4(x)
        return x

def calculate_time_dependent_metrics(surv_df, times, durations, events):
    """
    Calculate time-dependent concordance indices at specific time points.
    
    This function evaluates the model's predictive performance at specific timepoints
    by calculating the concordance index up to each time point.
    
    Args:
        surv_df (pd.DataFrame): Survival predictions DataFrame
        times (list): List of time points to evaluate (e.g., [30, 60, 90] days)
        durations (array): Array of observed durations
        events (array): Array of event indicators (1=event occurred, 0=censored)
    
    Returns:
        dict: Dictionary of time-specific concordance indices
    """
    time_specific_metrics = {}
    
    # Validate inputs
    if not isinstance(surv_df, pd.DataFrame):
        raise TypeError("surv_df must be a pandas DataFrame")
    
    for t in times:
        # For each time point, censor the data at that time point
        # This creates "artificial censoring" to evaluate performance up to time t
        censored_durations = np.minimum(durations, t)
        censored_events = events * (durations <= t)
        
        # Create EvalSurv object with censored data
        ev_t = EvalSurv(surv_df, censored_durations, censored_events, censor_surv='km')
        
        # Calculate concordance index using Antolini's method
        c_index_t = ev_t.concordance_td('antolini')
        time_specific_metrics[f'{t}_days'] = c_index_t
        
    return time_specific_metrics

def calculate_p_value(bootstrap_values, null_value=0.5):
    """
    Calculate two-sided p-value from bootstrap distribution.
    
    Tests whether the model's concordance index is significantly different from
    random prediction (null_value=0.5).
    
    Args:
        bootstrap_values (array): Array of bootstrap metric estimates
        null_value (float): Null hypothesis value (default: 0.5 for C-index)
    
    Returns:
        float: Two-sided p-value
    """
    # Calculate the mean of the bootstrap distribution
    observed = np.mean(bootstrap_values)
    
    # Calculate two-sided p-value based on bootstrap distribution
    if observed >= null_value:
        # If observed > null, calculate P(result ≤ null)
        p_value = 2 * (1 - np.mean(bootstrap_values >= null_value))
    else:
        # If observed < null, calculate P(result ≥ null)
        p_value = 2 * np.mean(bootstrap_values <= null_value)
    return p_value

def bootstrap_metric(surv_df, times, durations, events, n_bootstrap=1000, alpha=0.95):
    """
    Calculate time-dependent metrics with bootstrap confidence intervals and p-values.
    
    This function generates bootstrap samples of the test data to estimate
    confidence intervals and p-values for the time-dependent concordance indices.
    
    Args:
        surv_df (pd.DataFrame): Survival predictions DataFrame
        times (list): List of time points to evaluate
        durations (array): Array of observed durations
        events (array): Array of event indicators
        n_bootstrap (int): Number of bootstrap samples
        alpha (float): Confidence level for intervals (e.g., 0.95 for 95% CI)
    
    Returns:
        dict: Dictionary containing mean metrics, confidence intervals, and p-values
    """
    n_samples = len(durations)
    bootstrap_metrics = {f'{t}_days': [] for t in times}
    
    print(f"\nPerforming {n_bootstrap} bootstrap iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{n_bootstrap} iterations...")
            
        # Generate bootstrap sample indices (sampling with replacement)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Get bootstrap sample
        bootstrap_durations = durations[indices]
        bootstrap_events = events[indices]
        bootstrap_surv = surv_df.iloc[:, indices]
        
        # Calculate metrics for this bootstrap sample
        metrics = calculate_time_dependent_metrics(bootstrap_surv, times, bootstrap_durations, bootstrap_events)
        
        # Store results for each time point
        for t in times:
            bootstrap_metrics[f'{t}_days'].append(metrics[f'{t}_days'])
    
    # Calculate statistics for each time point
    results = {}
    for t in times:
        key = f'{t}_days'
        values = np.array(bootstrap_metrics[key])
        
        # Calculate mean, confidence interval, and p-value
        mean_value = np.mean(values)
        lower = np.percentile(values, (1 - alpha) * 100 / 2)  # Lower bound of CI
        upper = np.percentile(values, 100 - (1 - alpha) * 100 / 2)  # Upper bound of CI
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
    Pretty print metrics with confidence intervals and p-values.
    
    Formats and displays the time-dependent concordance indices with their
    confidence intervals and p-values in a readable format.
    
    Args:
        metrics_dict (dict): Dictionary containing mean metrics, CIs, and p-values
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
            # Format p-value for display
            p_value_str = f"p < 0.001" if metric['p_value'] < 0.001 else f"p = {metric['p_value']:.3f}"
            print(f"{label}: {metric['mean']:.2f} (95% CI: {metric['ci_lower']:.2f} - {metric['ci_upper']:.2f}), {p_value_str}")

def load_data(df, num_durations=50):
    """
    Load and preprocess data for the DeepHit model.
    
    This function:
    1. Splits data into train/validation/test sets
    2. Normalizes features using StandardScaler
    3. Prepares input and target variables in the required format
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and survival data
        num_durations (int): Number of discrete time intervals for the model
        
    Returns:
        tuple: Processed data ready for model training and evaluation
    """
    # Identify feature columns (all columns except duration and event)
    exclude_columns = ['duration', 'event']
    feature_cols = [col for col in df.columns if col not in exclude_columns]
    
    # Split into train/val/test with stratification by event status
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=44, 
        stratify=df['event']
    )
    
    df_train_val = df.loc[train_idx]
    df_test = df.loc[test_idx]
    
    train_idx_final, val_idx = train_test_split(
        df_train_val.index, test_size=1/8, random_state=44,
        stratify=df_train_val.loc[df_train_val.index, 'event']
    )
    
    df_val = df_train_val.loc[val_idx]
    df_train = df_train_val.loc[train_idx_final]

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[feature_cols])
    x_val = scaler.transform(df_val[feature_cols])
    x_test = scaler.transform(df_test[feature_cols])

    # Convert to float32 for better numerical stability in deep learning
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    # Return the processed data as a tuple
    return (x_train, x_val, x_test), \
           ((df_train['duration'].values, df_train['event'].values),
            (df_val['duration'].values, df_val['event'].values),
            (df_test['duration'].values, df_test['event'].values)), \
           (df_train, df_val, df_test)

def main():
    """
    Main function to train and evaluate the DeepHit model.
    
    This function:
    1. Loads and preprocesses the data
    2. Sets up and trains the DeepHit model
    3. Evaluates the model on training, validation, and test sets
    4. Calculates and reports various performance metrics
    5. Saves the model predictions for comparison with other models
    """
    # Set random seed for reproducibility
    set_seed(44)
    
    # Load and preprocess data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        script_dir, "../../Results/Results_v1/Results_0d_365d_window/bleeding_multihot_encoded.csv"
    )
    
    # Load the CSV file and preprocess the column names
    df = pd.read_csv(file_path)
    df = df[[col for col in df.columns if "set" not in col]]  # Remove columns with "set" in their name
    df = df.rename(columns={"tte": "duration", "Label": "event"})  # Rename columns to standard names
    df = df.drop(columns=["Pt_id"])  # Remove patient ID column
    
    # Ensure the event and duration columns are at the end of the DataFrame
    df = pd.concat([df.drop(columns=["event", "duration"]), df[["event", "duration"]]], axis=1)

    # Set the number of discrete time intervals for the DeepHit model
    num_durations = 23  # Increased for better granularity
    
    # Load and preprocess the data
    (x_train, x_val, x_test), \
    ((durations_train, events_train),
     (durations_val, events_val),
     (durations_test, events_test)), \
    (df_train, df_val, df_test) = load_data(df, num_durations)

    # Prepare label transform for discrete-time survival analysis
    labtrans = DeepHitSingle.label_transform(num_durations)
    y_train = labtrans.fit_transform(durations_train, events_train)
    y_val = labtrans.transform(durations_val, events_val)
    y_test = labtrans.transform(durations_test, events_test)

    # Set up the model architecture
    in_features = x_train.shape[1]  # Number of input features
    num_nodes = [256, 128, 64]  # Deeper architecture with decreasing neurons
    
    # Initialize custom network
    net = CustomNet(in_features, num_nodes, labtrans.out_features)
    
    # Initialize model with optimized hyperparameters
    optimizer = tt.optim.Adam(lr=0.001, weight_decay=0.001)
    model = DeepHitSingle(
        net, 
        optimizer=optimizer,
        alpha=0.15,  # Controls balance between likelihood and rank loss
        sigma=0.1,   # Controls smoothing of the ranking loss
        duration_index=labtrans.cuts  # Time intervals for discrete model
    )

    # Set training parameters
    batch_size = 128
    epochs = 200  # Higher value with early stopping to find optimal point
    
    # Set up callbacks for training
    callbacks = [
        tt.callbacks.EarlyStopping(
            patience=20,  # Stop if no improvement for 20 epochs
            min_delta=1e-4  # Minimum change to qualify as improvement
        )
    ]

    # Train the model
    print("\nTraining DeepHit model...")
    log = model.fit(
        x_train, y_train,
        batch_size,
        epochs,
        callbacks,
        val_data=(x_val, y_val),
        verbose=True
    )
   

    # Generate survival probability predictions
    surv_train_df = model.predict_surv_df(x_train)
    surv_val_df = model.predict_surv_df(x_val)
    surv_test_df = model.predict_surv_df(x_test)
    
    # Get time grid for evaluation metrics
    time_grid = model.duration_index

    # Define evaluation function for comprehensive model assessment
    def evaluate_predictions(surv_df, durations, events, dataset_name):
        """
        Evaluate model predictions with multiple metrics.
        
        Args:
            surv_df (pd.DataFrame): Survival predictions DataFrame
            durations (array): Observed durations
            events (array): Event indicators
            dataset_name (str): Name of the dataset for reporting (e.g., "Training", "Test")
            
        Returns:
            tuple: (concordance_td, integrated_brier_score, integrated_nbll)
        """
        # Create evaluation object
        ev = EvalSurv(
            surv_df,
            durations,
            events,
            censor_surv='km'  # Use Kaplan-Meier for censoring distribution
        )
        
        # Calculate overall survival analysis metrics
        concordance_td = ev.concordance_td('antolini')  # Time-dependent concordance index

        print(f"\n{dataset_name} Overall Performance Metrics:")
        print(f"Concordance TD: {concordance_td:.2f}")
       
        # Only calculate time-specific metrics for test set (to save computation)
        if dataset_name == "Test":
            # Calculate time-specific metrics at clinically relevant time points
            evaluation_times = [30, 60, 90, 180, 270, 365]  # Time points in days
            try:
                # Calculate point estimates
                time_specific_metrics = calculate_time_dependent_metrics(surv_df, evaluation_times, durations, events)
                
                print("\nPoint Estimates:")
                print(f"1 Month (30 days): {time_specific_metrics['30_days']:.2f}")
                print(f"2 Months (60 days): {time_specific_metrics['60_days']:.2f}")
                print(f"3 Months (90 days): {time_specific_metrics['90_days']:.2f}")
                print(f"6 Months (180 days): {time_specific_metrics['180_days']:.2f}")
                print(f"9 Months (270 days): {time_specific_metrics['270_days']:.2f}")
                print(f"12 Months (365 days): {time_specific_metrics['365_days']:.2f}")
                
                # Calculate bootstrapped estimates with confidence intervals and p-values
                bootstrapped_metrics = bootstrap_metric(surv_df, evaluation_times, durations, events, n_bootstrap=1000)
                print_metrics_with_ci_and_pvalue(bootstrapped_metrics)
                
            except Exception as e:
                print(f"Error calculating time-specific metrics: {str(e)}")
    
        return concordance_td

    # Evaluate model on all datasets
    train_metrics = evaluate_predictions(surv_train_df, durations_train, events_train, "Training")
    val_metrics = evaluate_predictions(surv_val_df, durations_val, events_val, "Validation")
    test_metrics = evaluate_predictions(surv_test_df, durations_test, events_test, "Test")

    # Save predictions for model comparison
    results = {
        'predictions': surv_test_df,
        'durations': durations_test,
        'events': events_test
    }

    # Create directory for results if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'model_comparison_results')
    os.makedirs(results_dir, exist_ok=True)

    # Save results to pickle file
    results_path = os.path.join(results_dir, 'deephit_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
