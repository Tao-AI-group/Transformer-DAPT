from collections import defaultdict
import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchvision.ops import sigmoid_focal_loss
from pycox.models.loss import NLLPCHazardLoss


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int, optional): How many epochs to wait after the last time
            validation loss improved before stopping. Default: 7.
        verbose (bool, optional): If True, prints a message each time the validation
            loss improves. Default: False.
        delta (float, optional): Minimum change in the monitored quantity to qualify as an
            improvement. Default: 0.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss: float, model: nn.Module, name: str = 'checkpoint.pt'):
        """
        Checks if validation loss improved. If not, increments the counter and 
        triggers early stopping if patience is exceeded.

        Args:
            val_loss (float): Current validation loss to check.
            model (nn.Module): Model instance to save if validation loss improved.
            name (str, optional): Name of the checkpoint file. Default: 'checkpoint.pt'.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, name: str):
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): New best validation loss.
            model (nn.Module): Model instance to save.
            name (str): Name of the checkpoint file.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss


class Trainer:
    """
    A trainer class for the Transformer_DAPT model, providing a training loop with
    OneCycleLR scheduling and optional early stopping for single-event survival tasks.

    Args:
        model (nn.Module): The Transformer_DAPT model to train.
        metrics (list, optional): List of metric/loss functions, where the first is
            assumed to be NLLPCHazardLoss. Default: [NLLPCHazardLoss(),].
    """
    def __init__(self, model: nn.Module, metrics=None):
        if metrics is None:
            metrics = [NLLPCHazardLoss()]
        self.model = model
        self.metrics = metrics

        # Data logging
        self.train_logs = defaultdict(list)

        # Utility function to extract survival targets (duration, event)
        self.get_target = lambda df: (df['duration'].values, df['event'].values)

        # GPU check
        self.use_gpu = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count()

        if self.use_gpu:
            print('Using CUDA for training.')
            if self.num_gpus > 1:
                # Wrap model in DataParallel for multi-GPU support
                self.model = nn.DataParallel(self.model)
                self.model_config = self.model.module.config
            else:
                self.model_config = self.model.config
            self.model.to("cuda")
        else:
            print('No GPU found, using CPU for training.')
            self.model_config = self.model.config

        self.early_stopping = None

        # Resolve checkpoint directory/path
        main_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        relative_ckpt_path = self.model_config['checkpoint']
        absolute_ckpt_path = os.path.normpath(os.path.join(main_script_dir, relative_ckpt_path))
        ckpt_dir = os.path.dirname(absolute_ckpt_path)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.ckpt = absolute_ckpt_path

    def train_single_event(
        self,
        train_set,
        val_set=None,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.01,
        val_batch_size: int = None,
        optimizer: str = "AdamW",
        **kwargs,
    ):
        """
        Train the Transformer_DAPT model for a single-event survival task using
        the AdamW optimizer and a OneCycleLR scheduler.

        Args:
            train_set (tuple): ((x_train, mask_train), df_y_train).
            val_set (tuple, optional): ((x_val, mask_val), df_y_val). Default: None.
            batch_size (int, optional): Training batch size. Default: 32.
            epochs (int, optional): Number of training epochs. Default: 100.
            learning_rate (float, optional): Initial (max) learning rate. Default: 0.0003.
            weight_decay (float, optional): Weight decay for AdamW. Default: 0.01.
            val_batch_size (int, optional): Validation batch size. If None, uses `batch_size`. Default: None.
            optimizer (str, optional): Optimizer name. Currently only "AdamW" is supported. Default: "AdamW".
            **kwargs: Additional keyword arguments for training logic.

        Returns:
            tuple: (train_loss_list, val_loss_list)
        """
        # Unpack training data
        (x_train, mask_train), df_y_train = train_set
        tensor_y_train = torch.tensor(df_y_train.values, dtype=torch.float32)
        tensor_train = x_train.float()
        tensor_mask_train = mask_train.float()

        if val_set is not None:
            (x_val, mask_val), df_y_val = val_set
            tensor_val = x_val.float()
            tensor_mask_val = mask_val.float()
            tensor_y_val = torch.tensor(df_y_val.values, dtype=torch.float32)
            self.early_stopping = EarlyStopping(patience=self.model_config['early_stop_patience'])

        # Group parameters for weight decay
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Set up OneCycleLR scheduler
        num_training_steps = epochs * int(np.ceil(len(df_y_train) / batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=num_training_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1e4
        )

        # Additional losses
        Triplet_loss = nn.TripletMarginLoss()
        BCE_loss = None if self.model_config['focal'] else nn.BCEWithLogitsLoss()

        num_train_batches = int(np.ceil(len(df_y_train) / batch_size))
        train_loss_list, val_loss_list = [], []

        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_Hazard = 0.0
            epoch_Triplet = 0.0
            epoch_BCE = 0.0

            self.model.train()

            for batch_idx in range(num_train_batches):
                optimizer.zero_grad()

                # Create batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_x = tensor_train[start_idx:end_idx].long()
                batch_mask = tensor_mask_train[start_idx:end_idx]
                batch_y_train = tensor_y_train[start_idx:end_idx]

                if self.use_gpu:
                    batch_x = batch_x.to("cuda", non_blocking=True)
                    batch_mask = batch_mask.to("cuda", non_blocking=True)
                    batch_y_train = batch_y_train.to("cuda", non_blocking=True)

                # Forward pass
                phi = self.model((batch_x, batch_mask))

                # Potential triplet sampling
                triplet_indicator = False
                triplet_number = self.model_config["triplet_size"]
                class_counts = torch.bincount(batch_y_train[:, 1].long())
                if (class_counts >= triplet_number).all():
                    neg_idx = (batch_y_train[:, 1] == 0).nonzero(as_tuple=True)[0].tolist()
                    pos_idx = (batch_y_train[:, 1] == 1).nonzero(as_tuple=True)[0].tolist()

                    seleted_neg = random.sample(neg_idx, triplet_number)
                    seleted_pos = random.sample(pos_idx, triplet_number)

                    neg_mean = phi[0][neg_idx].mean(dim=0).repeat(triplet_number, 1)
                    pos_mean = phi[0][pos_idx].mean(dim=0).repeat(triplet_number, 1)

                    anchor = phi[0][seleted_neg + seleted_pos]
                    negative = torch.cat((pos_mean, neg_mean), 0)
                    positive = torch.cat((neg_mean, pos_mean), 0)

                    triplet_indicator = True

                # Compute losses
                if len(self.metrics) == 1:
                    # Hazard loss
                    batch_Hazard = self.metrics[0](
                        phi[1],
                        batch_y_train[:, 0].long(),
                        batch_y_train[:, 1].long(),
                        batch_y_train[:, 2].float()
                    )

                    batch_Triplet = 0.0
                    if triplet_indicator:
                        batch_Triplet = Triplet_loss(anchor, positive, negative)

                    # BCE or Focal
                    if self.model_config['focal']:
                        batch_BCE = sigmoid_focal_loss(
                            phi[3].flatten(),
                            batch_y_train[:, 1].float(),
                            self.model_config['class_weight'],
                            reduction="mean"
                        )
                    else:
                        batch_BCE = BCE_loss(phi[3].flatten(), batch_y_train[:, 1].float())

                    # Weighted total loss
                    batch_loss = (self.model_config["Hazard_weight"] * batch_Hazard
                                  + self.model_config["BCE_weight"] * batch_BCE)
                    if triplet_indicator:
                        batch_loss += self.model_config["Triplet_weight"] * batch_Triplet

                    # Accumulate for logging
                    epoch_loss += batch_loss.item()
                    epoch_Hazard += batch_Hazard.item()
                    epoch_BCE += batch_BCE.item()
                    if triplet_indicator:
                        epoch_Triplet += batch_Triplet.item()
                else:
                    raise NotImplementedError("Multiple metrics not implemented in this trainer.")

                # Gradient clipping
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Backprop and step
                batch_loss.backward()
                optimizer.step()
                scheduler.step()

            # Average losses
            train_loss_list.append(epoch_loss / num_train_batches)
            epoch_loss /= num_train_batches
            epoch_Hazard /= num_train_batches
            epoch_Triplet /= num_train_batches
            epoch_BCE /= num_train_batches

            # Validation
            if val_set is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_num_batches = int(np.ceil(len(tensor_val) / (val_batch_size or batch_size)))
                    for val_batch_idx in range(val_num_batches):
                        val_start = val_batch_idx * (val_batch_size or batch_size)
                        val_end = val_start + (val_batch_size or batch_size)
                        val_batch_x = tensor_val[val_start:val_end].long()
                        val_batch_mask = tensor_mask_val[val_start:val_end]
                        val_batch_y = tensor_y_val[val_start:val_end]

                        if self.use_gpu:
                            val_batch_x = val_batch_x.to("cuda")
                            val_batch_mask = val_batch_mask.to("cuda")
                            val_batch_y = val_batch_y.to("cuda")

                        phi_val = self.model((val_batch_x, val_batch_mask))
                        hidden_state, predict_logits, embeddings, binary_logits = phi_val

                        # Compute validation hazard loss
                        val_Hazard = self.metrics[0](
                            predict_logits,
                            val_batch_y[:, 0].long(),
                            val_batch_y[:, 1].long(),
                            val_batch_y[:, 2].float()
                        )
                        val_loss += val_Hazard.item()

                    val_loss /= val_num_batches
                    val_loss_list.append(val_loss)

                    # Print progress
                    print(f"[Train-{epoch}] Total Loss: {epoch_loss:.4f}, "
                          f"Hazard Loss: {epoch_Hazard:.4f}, BCE Loss: {epoch_BCE:.4f}, "
                          f"Triplet Loss: {epoch_Triplet:.4f}")
                    print(f"[Val-{epoch}] Total Loss: {val_loss:.4f}")

                    # Early stopping
                    self.early_stopping(val_loss, self.model, name=self.ckpt)
                    if self.early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        # Load the best checkpoint
                        self.model.load_state_dict(torch.load(self.ckpt, weights_only=True))
                        return train_loss_list, val_loss_list

            else:
                print(f"[Train-{epoch}] Total Loss: {epoch_loss:.4f}, "
                      f"Hazard Loss: {epoch_Hazard:.4f}, BCE Loss: {epoch_BCE:.4f}, "
                      f"Triplet Loss: {epoch_Triplet:.4f}")

        if val_set is not None:
            print("Loading best model from checkpoint...")
            self.model.load_state_dict(torch.load(self.ckpt, weights_only=True))

        return train_loss_list, val_loss_list

    def fit(
        self,
        train_set,
        val_set=None,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
        val_batch_size: int = None,
        optimizer=None,
        **kwargs,
    ):
        """
        Main entry point for training. This trainer only supports single-event survival tasks.

        Args:
            train_set (tuple): The training dataset ((x_train, mask_train), df_y_train).
            val_set (tuple, optional): The validation dataset ((x_val, mask_val), df_y_val). Default: None.
            batch_size (int, optional): Training batch size. Default: 128.
            epochs (int, optional): Number of training epochs. Default: 100.
            learning_rate (float, optional): Learning rate for optimizer. Default: 1e-3.
            weight_decay (float, optional): Weight decay for optimizer. Default: 0.
            val_batch_size (int, optional): Validation batch size. Default: None.
            optimizer (str, optional): Optimizer choice. Default: None (internal default used).
            **kwargs: Additional keyword arguments passed to the underlying training method.

        Returns:
            tuple: (train_loss_list, val_loss_list)
        """
        # This trainer only supports single-event tasks.
        if self.model_config.num_event != 1:
            raise ValueError(
                f"Trainer only supports single-event tasks. Found num_event={self.model_config.num_event}"
            )

        return self.train_single_event(
            train_set=train_set,
            val_set=val_set,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_batch_size=val_batch_size,
            optimizer=optimizer,
            **kwargs,
        )
