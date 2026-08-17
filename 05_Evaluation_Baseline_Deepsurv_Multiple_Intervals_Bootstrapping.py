import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
from torch import nn, optim
from pycox.models.loss import CoxPHLoss
import warnings
warnings.filterwarnings('ignore')

# Set GPU device if available
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed):
    """Set random seed for reproducibility across numpy, torch, and cuda"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_target(df):
    """Extract duration and event information from dataframe"""
    return (df['duration'].values, df['event'].values)

class DeepSurvNet(nn.Module):
    """
    Neural network architecture for DeepSurv model
    - Three fully connected layers with ReLU activation
    - Batch normalization and dropout for regularization
    """
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # Initialize weights for better convergence
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle edge case of single sample by duplicating it
        # (BatchNorm requires more than one sample)
        if x.size(0) == 1:
            x = x.repeat(2, 1)
        return self.net(x)

class DeepSurv:
    """
    DeepSurv implementation based on Cox Proportional Hazards loss
    """
    def __init__(self, net, optimizer=None, device=None):
        self.net = net
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        self.optimizer = optimizer or optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=0.01, eps=1e-8)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.loss_fn = CoxPHLoss()
        self.baseline_hazards_ = None
        self.durations = None
        
    def fit(self, input, batch_size, epochs, callbacks=None, verbose=True, val_data=None):
        """
        Train the DeepSurv model
        
        Args:
            input: Tuple of (x, (durations, events))
            batch_size: Size of mini-batches for training
            epochs: Maximum number of training epochs
            callbacks: Optional callback functions (not used)
            verbose: Whether to print progress
            val_data: Optional validation data for early stopping
        """
        x, (durations, events) = input
        if val_data:
            val_x, (val_durations, val_events) = val_data
            
        # Convert inputs to torch tensors and move to device
        x = torch.FloatTensor(x).to(self.device)
        durations = torch.FloatTensor(durations).to(self.device)
        events = torch.FloatTensor(events).to(self.device)
        
        if val_data:
            val_x = torch.FloatTensor(val_x).to(self.device)
            val_durations = torch.FloatTensor(val_durations).to(self.device)
            val_events = torch.FloatTensor(val_events).to(self.device)
        
        # Ensure batch size is at least 2 for BatchNorm
        batch_size = max(batch_size, 2)
        n_batches = int(np.ceil(len(x) / batch_size))
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0
            valid_batches = 0
            indices = torch.randperm(len(x))
            
            for i in range(n_batches):
                try:
                    batch_idx = indices[i*batch_size:min((i+1)*batch_size, len(x))]
                    
                    # Skip batches that are too small
                    if len(batch_idx) < 2:
                        continue
                    
                    self.optimizer.zero_grad()
                    batch_x = x[batch_idx]
                    batch_durations = durations[batch_idx]
                    batch_events = events[batch_idx]
                    
                    # Forward pass with clipping to prevent numerical instability
                    predictions = self.net(batch_x)
                    predictions = torch.clamp(predictions, -10, 10)
                    
                    try:
                        # Calculate Cox PH loss
                        loss = self.loss_fn(predictions, batch_durations, batch_events)
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        # Backpropagation with gradient clipping
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                        self.optimizer.step()
                        epoch_loss += loss.item()
                        valid_batches += 1
                        
                    except RuntimeError:
                        continue
                    
                except Exception:
                    continue
            
            if valid_batches > 0:
                epoch_loss /= valid_batches
                
                # Validation and early stopping if validation data provided
                if val_data:
                    self.net.eval()
                    with torch.no_grad():
                        val_pred = []
                        val_losses = []
                        for i in range(0, len(val_x), batch_size):
                            batch = val_x[i:min(i+batch_size, len(val_x))]
                            if len(batch) < 2:
                                continue
                            batch_durations = val_durations[i:min(i+batch_size, len(val_x))]
                            batch_events = val_events[i:min(i+batch_size, len(val_x))]
                            
                            pred = self.net(batch)
                            pred = torch.clamp(pred, -10, 10)
                            try:
                                batch_loss = self.loss_fn(pred, batch_durations, batch_events)
                                if not torch.isnan(batch_loss) and not torch.isinf(batch_loss):
                                    val_losses.append(batch_loss.item())
                                    val_pred.append(pred)
                            except:
                                continue
                        
                        if val_losses:
                            val_loss = np.mean(val_losses)
                            
                            # Learning rate scheduling based on validation loss
                            self.scheduler.step(val_loss)
                            
                            # Save model if validation loss improves
                            if val_loss < best_loss:
                                best_loss = val_loss
                                patience_counter = 0
                                best_model_state = self.net.state_dict()
                            else:
                                patience_counter += 1
                            
                            if verbose:
                                print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.2f} - Val Loss: {val_loss:.2f}')
                            
                            # Early stopping after 15 epochs without improvement
                            if patience_counter >= 15:
                                print('Early stopping triggered')
                                self.net.load_state_dict(best_model_state)
                                break
                elif verbose:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.2f}')

        # Load best model if we found one during validation
        if best_model_state is not None:
            self.net.load_state_dict(best_model_state)
        return self

    def predict(self, input):
        """
        Generate risk predictions for input data
        
        Args:
            input: Features for prediction
            
        Returns:
            numpy array of predictions
        """
        self.net.eval()
        with torch.no_grad():
            if not isinstance(input, torch.Tensor):
                input = torch.FloatTensor(input).to(self.device)
            
            predictions = []
            batch_size = 32
            for i in range(0, len(input), batch_size):
                batch = input[i:min(i+batch_size, len(input))]
                # Handle single sample case
                if len(batch) == 1:
                    batch = batch.repeat(2, 1)
                pred = self.net(batch)
                pred = torch.clamp(pred, -100, 100)
                predictions.append(pred[:1] if len(batch) == 2 and i+1 >= len(input) else pred)
            
            return torch.cat(predictions, 0).cpu().numpy()

    def compute_baseline_hazards(self, input):
        """
        Compute baseline hazards for survival function estimation
        
        Args:
            input: Tuple of (x, (durations, events))
            
        Returns:
            self with computed baseline hazards
        """
        x, (durations, events) = input
        x = torch.FloatTensor(x).to(self.device)
        self.durations = np.unique(durations)
        
        risk_scores = np.exp(self.predict(x))
        self.baseline_hazards_ = {}
        for t in self.durations:
            mask = (durations == t) & (events == 1)
            if mask.any():
                at_risk = (durations >= t)
                hazard = mask.sum() / (risk_scores[at_risk].sum() + 1e-8)
                self.baseline_hazards_[t] = hazard
        
        return self

    def predict_surv(self, input):
        """
        Predict survival function for input data
        
        Args:
            input: Features for prediction
            
        Returns:
            Survival probability matrix (samples x timepoints)
        """
        if self.baseline_hazards_ is None:
            raise ValueError('Need to compute baseline hazards first.')
            
        risk_scores = np.clip(np.exp(self.predict(input)), 1e-8, 1e8)
        surv = np.zeros((len(input), len(self.durations)))
        
        cum_hazard = np.zeros(len(self.durations))
        for i, t in enumerate(self.durations):
            if t in self.baseline_hazards_:
                cum_hazard[i] = self.baseline_hazards_[t]
        cum_hazard = np.cumsum(cum_hazard)
        
        for i in range(len(input)):
            surv[i] = np.exp(-risk_scores[i] * cum_hazard)
        
        return surv

    def predict_surv_df(self, input):
        """
        Predict survival function as a DataFrame
        
        Args:
            input: Features for prediction
            
        Returns:
            DataFrame with survival probabilities
        """
        surv = self.predict_surv(input)
        return pd.DataFrame(surv.T, index=self.durations)

def calculate_time_dependent_metrics(surv_df, times, durations, events):
    """
    Calculate time-dependent concordance indices at specific time points.
    
    Args:
        surv_df: Survival predictions DataFrame
        times: List of time points to evaluate
        durations: Array of observed durations
        events: Array of event indicators
    
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
    observed = np.mean(bootstrap_values)
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
        n_bootstrap: Number of bootstrap iterations
        alpha: Confidence level
        
    Returns:
        Dictionary of metrics with confidence intervals and p-values
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
        
        mean_value = np.mean(values)
        lower = np.percentile(values, (1 - alpha) * 100 / 2)
        upper = np.percentile(values, 100 - (1 - alpha) * 100 / 2)
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
    
    Args:
        metrics_dict: Dictionary of metrics with confidence intervals and p-values
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


def load_data(df):
    """
    Split data into train, validation and test sets with feature scaling
    
    Args:
        df: DataFrame with features and targets
        
    Returns:
        Tuple of (feature arrays, target arrays, dataframes)
    """
    exclude_columns = ['duration', 'event']
    feature_cols = [col for col in df.columns if col not in exclude_columns]
    
    # Split into train+val and test (80/20)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=44, 
        stratify=df.loc[df.index, 'event']
    )
    
    df_train_val = df.loc[train_idx]
    df_test = df.loc[test_idx]
    
    # Split train+val into train and validation (7/1 ratio)
    train_idx_final, val_idx = train_test_split(
        df_train_val.index, test_size=1/8, random_state=44,
        stratify=df_train_val.loc[df_train_val.index, 'event']
    )
    
    df_val = df_train_val.loc[val_idx]
    df_train = df_train_val.loc[train_idx_final]

    print("\nData Split Statistics:")
    print(f"Training set size: {len(df_train)}, Event rate: {df_train['event'].mean():.3f}")
    print(f"Validation set size: {len(df_val)}, Event rate: {df_val['event'].mean():.3f}")
    print(f"Test set size: {len(df_test)}, Event rate: {df_test['event'].mean():.3f}")

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df_train[feature_cols])
    x_val = scaler.transform(df_val[feature_cols])
    x_test = scaler.transform(df_test[feature_cols])

    print("\nFeature Statistics:")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Training input shape: {x_train.shape}")
    print(f"Validation input shape: {x_val.shape}")
    print(f"Test input shape: {x_test.shape}")

    # Convert to float32 for GPU efficiency
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, x_val, x_test), \
           (get_target(df_train), get_target(df_val), get_target(df_test)), \
           (df_train, df_val, df_test)


def main():
    # Set random seed for reproducibility
    set_seed(44)
    
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        script_dir, "../../Results/Results_v1/Results_0d_365d_window/bleeding_multihot_encoded.csv"
    )
    print(f"Loading data from: {file_path}")

    # Load and preprocess data
    df = pd.read_csv(file_path)
    df = df[[col for col in df.columns if "set" not in col]]
    df = df.rename(columns={"tte": "duration", "Label": "event"})
    df = df.drop(columns=["Pt_id"])
    df = pd.concat([df.drop(columns=["event", "duration"]), df[["event", "duration"]]], axis=1)

    print("\nDataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Event rate: {df['event'].mean():.3f}")
    print(f"Duration statistics:")
    print(df['duration'].describe())

    # Split data and preprocess features
    (x_train, x_val, x_test), \
    ((y_train_duration, y_train_event), 
        (y_val_duration, y_val_event),
        (y_test_duration, y_test_event)), \
    (df_train, df_val, df_test) = load_data(df)

    # Initialize model
    print("\nInitializing DeepSurv model...")
    in_features = x_train.shape[1]
    net = DeepSurvNet(in_features)
    model = DeepSurv(
        net,
        optimizer=optim.Adam(
            net.parameters(), 
            lr=0.0001,
            weight_decay=0.01
        )
    )

    # Training hyperparameters
    batch_size = 32
    epochs = 100

    print("\nTraining DeepSurv model...")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print(f"Learning rate: 0.0001")
    print(f"Using device: {model.device}")

    # Prepare data for training
    train_data = (x_train, (y_train_duration, y_train_event))
    val_data = (x_val, (y_val_duration, y_val_event))

    # Train model
    model.fit(
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        val_data=val_data
    )

    # Evaluate on training set
    print("\nEvaluating on training set...")
    model.compute_baseline_hazards(train_data)
    surv_train = model.predict_surv_df(x_train)
    ev_train = EvalSurv(surv_train, y_train_duration, y_train_event, censor_surv='km')
    train_c_index = ev_train.concordance_td('antolini')
    print(f"Training Concordance Index: {train_c_index:.2f}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    surv_val = model.predict_surv_df(x_val)
    ev_val = EvalSurv(surv_val, y_val_duration, y_val_event, censor_surv='km')
    val_c_index = ev_val.concordance_td('antolini')
    print(f"Validation Concordance Index: {val_c_index:.2f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    surv_test = model.predict_surv_df(x_test)

    # Save predictions and test data
    results = {
        'predictions': surv_test,
        'durations': y_test_duration,
        'events': y_test_event
    }
    
    # Create results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'model_comparison_results')
    os.makedirs(results_dir, exist_ok=True)

    # Save results
    results_path = os.path.join(results_dir, 'deepsurv_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    # Calculate test concordance index
    ev_test = EvalSurv(surv_test, y_test_duration, y_test_event, censor_surv='km')
    test_c_index = ev_test.concordance_td('antolini')
    print(f"Test Concordance Index: {test_c_index:.2f}")

     # Calculate risk scores and binary metrics for test set
    test_risk_scores = -np.log(surv_test.iloc[-1].values)
    test_predictions = (test_risk_scores > np.median(test_risk_scores)).astype(float)
    test_accuracy = accuracy_score(y_test_event, test_predictions)
    test_f1 = f1_score(y_test_event, test_predictions)
    test_auroc = roc_auc_score(y_test_event, test_risk_scores)
    test_auprc = average_precision_score(y_test_event, test_risk_scores)
    test_precision = precision_score(y_test_event, test_predictions)
    test_sensitivity = recall_score(y_test_event, test_predictions)
    tn, fp, fn, tp = confusion_matrix(y_test_event, test_predictions).ravel()
    test_specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 score: {test_f1:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Sensitivity (Recall): {test_sensitivity:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")

    # Calculate time-specific metrics with bootstrapping
    print("\nCalculating time-specific metrics...")
    evaluation_times = [30, 60, 90, 180, 270, 365]  # Time points in days
    try:
        # Calculate point estimates
        time_specific_metrics = calculate_time_dependent_metrics(surv_test, evaluation_times, y_test_duration, y_test_event)
        
        print("\nPoint Estimates:")
        print(f"1 Month (30 days): {time_specific_metrics['30_days']:.2f}")
        print(f"2 Months (60 days): {time_specific_metrics['60_days']:.2f}")
        print(f"3 Months (90 days): {time_specific_metrics['90_days']:.2f}")
        print(f"6 Months (180 days): {time_specific_metrics['180_days']:.2f}")
        print(f"9 Months (270 days): {time_specific_metrics['270_days']:.2f}")
        print(f"12 Months (365 days): {time_specific_metrics['365_days']:.2f}")
        
        # Calculate bootstrapped estimates with confidence intervals and p-values
        bootstrapped_metrics = bootstrap_metric(surv_test, evaluation_times, y_test_duration, y_test_event, n_bootstrap=1000)
        print_metrics_with_ci_and_pvalue(bootstrapped_metrics)
        
    except Exception as e:
        print(f"Error calculating time-specific metrics: {str(e)}")

if __name__ == "__main__":
    main()
