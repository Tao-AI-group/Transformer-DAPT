import numpy as np
from typing import List, Dict
import pandas as pd

class ModelComparison:
    """
    Utility class for statistical comparison of survival models using bootstrap.
    """
    def __init__(self, n_bootstrap: int = 1000, alpha: float = 0.05):
        """
        Args:
            n_bootstrap: Number of bootstrap resamples.
            alpha: Significance level for confidence intervals (e.g., 0.05 for 95% CI).
        """
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        
    def bootstrap_model_comparison(
        self,
        surv_df1: pd.DataFrame,
        surv_df2: pd.DataFrame,
        durations: np.ndarray,
        events: np.ndarray,
        times: List[int],
        model1_name: str = "Model1",
        model2_name: str = "Model2"
    ) -> Dict:
        """
        Perform a bootstrap comparison between two survival models at specified timepoints.

        Args:
            surv_df1: Survival predictions (as a DataFrame) from the first model.
            surv_df2: Survival predictions (as a DataFrame) from the second model.
            durations: Array of observed durations (time to event or censoring).
            events: Array of event indicators (1 if event occurred, 0 if censored).
            times: List of timepoints at which to compare performance.
            model1_name: Name/identifier of the first model.
            model2_name: Name/identifier of the second model.

        Returns:
            A dictionary containing:
                - model_names: Tuple of (model1_name, model2_name).
                - results: Dictionary where each key corresponds to a timepoint (e.g., '30_days'),
                  and each value contains mean_difference, confidence intervals, and p_value.
        """
        # Number of samples in the original dataset
        n_samples = len(durations)
        
        # Dictionary to accumulate c-index differences for each timepoint across bootstrap iterations
        performance_diff = {f'{t}_days': [] for t in times}
        
        print(f"\nPerforming {self.n_bootstrap} bootstrap iterations for model comparison...")
        
        # Main loop for bootstrap resampling
        for i in range(self.n_bootstrap):
            # Progress update every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.n_bootstrap} iterations...")
                
            # Generate bootstrap sample by sampling indices with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_durations = durations[indices]
            bootstrap_events = events[indices]
            
            # Select the corresponding survival predictions for both models
            bootstrap_surv1 = surv_df1.iloc[:, indices]
            bootstrap_surv2 = surv_df2.iloc[:, indices]
            
            # Calculate performance (time-specific c-index) for each specified timepoint
            for t in times:
                c_index1 = self._calculate_time_specific_cindex(
                    bootstrap_surv1, t, bootstrap_durations, bootstrap_events)
                c_index2 = self._calculate_time_specific_cindex(
                    bootstrap_surv2, t, bootstrap_durations, bootstrap_events)
                
                # Store the difference in c-index (model1 minus model2)
                performance_diff[f'{t}_days'].append(c_index1 - c_index2)
        
        # Calculate mean difference, confidence intervals, and p-values for each timepoint
        results = {}
        for t in times:
            key = f'{t}_days'
            diff_values = np.array(performance_diff[key])
            
            # Mean difference
            mean_diff = np.mean(diff_values)
            
            # Confidence intervals based on alpha
            ci_lower = np.percentile(diff_values, (self.alpha / 2) * 100)
            ci_upper = np.percentile(diff_values, (1 - self.alpha / 2) * 100)
            
            # Two-sided p-value (H0: difference = 0)
            p_value = 2 * min(
                np.mean(diff_values >= 0),
                np.mean(diff_values <= 0)
            )
            
            # Store results in a dictionary
            results[key] = {
                'mean_difference': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value
            }
        
        return {
            'model_names': (model1_name, model2_name),
            'results': results
        }
    
    def _calculate_time_specific_cindex(
        self,
        surv_df: pd.DataFrame,
        t: int,
        durations: np.ndarray,
        events: np.ndarray
    ) -> float:
        """
        Calculate the time-specific C-index for a single timepoint using pycox's EvalSurv.
        
        Args:
            surv_df: DataFrame containing survival functions for a set of individuals.
            t: Specific timepoint at which to evaluate the c-index.
            durations: Array of observed durations (time to event or censoring).
            events: Array of event indicators (1 if event occurred, 0 if censored).
        
        Returns:
            The time-dependent concordance index (Antolini's c-index) at time t.
        """
        from pycox.evaluation import EvalSurv
        
        # Censor durations at time t
        censored_durations = np.minimum(durations, t)
        # Only consider events that occurred on or before time t
        censored_events = events * (durations <= t)
        
        # Create EvalSurv object for c-index calculation
        ev = EvalSurv(surv_df, censored_durations, censored_events, censor_surv='km')
        return ev.concordance_td('antolini')
    
    def print_comparison_results(self, comparison_results: Dict):
        """
        Pretty-print the comparison results.
        
        Args:
            comparison_results: Dictionary containing results from `bootstrap_model_comparison`.
        """
        model1_name, model2_name = comparison_results['model_names']
        results = comparison_results['results']
        
        print(f"\nStatistical Comparison Results: {model1_name} vs {model2_name}")
        print("Positive differences indicate better performance by", model1_name)
        print("Negative differences indicate better performance by", model2_name)
        
        # Optional mapping from time keys to descriptive labels
        time_mapping = {
            '30_days': '1 Month (30 days)',
            '60_days': '2 Months (60 days)',
            '90_days': '3 Months (90 days)',
            '180_days': '6 Months (180 days)',
            '270_days': '9 Months (270 days)',
            '365_days': '12 Months (365 days)'
        }
        
        for key, label in time_mapping.items():
            if key in results:
                metric = results[key]
                diff = metric['mean_difference']
                
                # Format p-value string
                p_value_str = "p < 0.001" if metric['p_value'] < 0.001 else f"p = {metric['p_value']:.3f}"
                
                print(f"\n{label}:")
                print(f"Mean Difference: {diff:.4f}")
                print(f"95% CI: [{metric['ci_lower']:.4f}, {metric['ci_upper']:.4f}]")
                print(f"Statistical Significance: {p_value_str}")
                
                # Interpretation
                if metric['p_value'] < 0.05:
                    better_model = model1_name if diff > 0 else model2_name
                    print(f"Significant difference in favor of {better_model}")
                else:
                    print("No statistically significant difference between models")
