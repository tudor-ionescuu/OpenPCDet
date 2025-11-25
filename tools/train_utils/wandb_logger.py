"""
Weights & Biases Logger for OpenPCDet training
"""
import os
import wandb
from pathlib import Path


class WandbLogger:
    def __init__(self, cfg, args, logger, enabled=True):
        """
        Initialize wandb logger
        
        Args:
            cfg: config object
            args: argparse arguments
            logger: python logger
            enabled: whether to enable wandb logging
        """
        self.enabled = enabled
        self.logger = logger
        
        if not self.enabled:
            self.logger.info("Wandb logging disabled")
            return
            
        try:
            # Initialize wandb
            wandb.init(
                project=args.wandb_project if hasattr(args, 'wandb_project') else "pointpillars-waymo",
                name=f"{cfg.TAG}_{args.extra_tag}",
                config={
                    "model": cfg.TAG,
                    "dataset": cfg.DATA_CONFIG._BASE_CONFIG_,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "optimizer": cfg.OPTIMIZATION.OPTIMIZER,
                    "learning_rate": cfg.OPTIMIZATION.LR,
                    "weight_decay": cfg.OPTIMIZATION.WEIGHT_DECAY,
                    **vars(args)
                },
                tags=[cfg.TAG, args.extra_tag],
                notes=f"Training {cfg.TAG} on Waymo dataset"
            )
            
            # Watch model (optional - can be memory intensive)
            # wandb.watch(model, log="all", log_freq=100)
            
            self.logger.info(f"Wandb initialized: {wandb.run.name}")
            self.logger.info(f"Wandb URL: {wandb.run.url}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.enabled = False
    
    def log_metrics(self, metrics_dict, step, epoch=None):
        """Log metrics to wandb"""
        if not self.enabled:
            return
        try:
            # Add epoch to metrics if provided
            if epoch is not None:
                metrics_dict['epoch'] = epoch
            wandb.log(metrics_dict, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log to wandb: {e}")
    
    def log_epoch_metrics(self, epoch, train_loss, lr=None, additional_metrics=None):
        """Log epoch-level metrics"""
        if not self.enabled:
            return
        
        metrics = {
            "epoch": epoch,
            "epoch_summary/train_loss": train_loss,
        }
        
        if lr is not None:
            metrics["epoch_summary/learning_rate"] = lr
        
        if additional_metrics:
            for key, val in additional_metrics.items():
                metrics[f"epoch_summary/{key}"] = val
        
        try:
            wandb.log(metrics)
            self.logger.info(f"Epoch {epoch} metrics logged to wandb")
        except Exception as e:
            self.logger.warning(f"Failed to log epoch metrics: {e}")
    
    def finish(self):
        """Finish wandb run"""
        if not self.enabled:
            return
        try:
            wandb.finish()
            self.logger.info("Wandb run finished")
        except Exception as e:
            self.logger.warning(f"Failed to finish wandb: {e}")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0.0, mode='min', logger=None):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            logger: python logger
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.logger = logger
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if self.logger:
            self.logger.info(f"Early stopping initialized with patience={patience}, min_delta={min_delta}, mode={mode}")
    
    def __call__(self, metric, epoch):
        """
        Check if training should stop
        
        Args:
            metric: validation metric value (loss or accuracy)
            epoch: current epoch number
            
        Returns:
            bool: True if training should stop
        """
        score = -metric if self.mode == 'min' else metric
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.logger:
                self.logger.info(f"Epoch {epoch}: Initial best score = {metric:.4f}")
            return False
        
        if score > self.best_score + self.min_delta:
            # Improvement
            if self.logger:
                self.logger.info(f"Epoch {epoch}: Score improved from {-self.best_score if self.mode == 'min' else self.best_score:.4f} "
                               f"to {metric:.4f}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.logger:
                self.logger.info(f"Epoch {epoch}: No improvement for {self.counter}/{self.patience} epochs "
                               f"(current: {metric:.4f}, best: {-self.best_score if self.mode == 'min' else self.best_score:.4f} at epoch {self.best_epoch})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.logger:
                    self.logger.info(f"Early stopping triggered! Best score {-self.best_score if self.mode == 'min' else self.best_score:.4f} at epoch {self.best_epoch}")
                return True
            
            return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
