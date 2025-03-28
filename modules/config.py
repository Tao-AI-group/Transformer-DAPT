from easydict import EasyDict

STConfig = EasyDict({
    # Number of discrete intervals for prediction. For example, if this is 5,
    # the entire observed period is discretized into 5 intervals.
    'num_durations': 5,

    # File path where model checkpoints will be saved or loaded.
    'checkpoint': 'checkpoints/MHA_Bleeding_Surv.pt',

    # Dimensionality of hidden embeddings, particularly in the attention layer.
    'hidden_size': 16,

    # The sample size for triplet-margin loss (if used).
    'triplet_size': 8,

    # Subdivision factor for hazard predictions (PC-Hazard approach). Usually 1 if not subdividing intervals further.
    'sub': 1,

    # Number of hidden layers (transformer blocks).
    'num_hidden_layers': 1,

    # Dropout probability for fully connected layers and/or embeddings.
    'hidden_dropout_prob': 0.3,

    # Total number of covariates/features for each patient (numerical + categorical).
    'num_feature': 9,

    # Number of categorical covariates for each patient. Used to set up feature embedding layers.
    'num_categorical_feature': 4,

    # Uncomment if needed for multi-dimensional output. E.g., multi-horizon survival predictions:
    # 'out_feature': 3,

    # Number of competing events. Set to 1 for single-event survival analysis.
    'num_event': 1,

    # Name of the activation function used within transformer blocks (e.g., 'gelu', 'relu').
    'hidden_act': 'gelu',

    # Dropout probability for attention weights.
    'attention_probs_dropout_prob': 0.1,

    # Patience for early stopping. The training ends if validation loss doesn't improve after these many epochs.
    'early_stop_patience': 20,

    # Range for the uniform or normal parameter initialization.
    'initializer_range': 0.001,

    # 'layer_norm_eps': 1e-12,  # Uncomment or set if needed, e.g., for layer normalization
})
