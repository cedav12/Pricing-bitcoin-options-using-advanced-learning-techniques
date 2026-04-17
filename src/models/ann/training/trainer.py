import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from collections import defaultdict

class Trainer:
    """Orchestrates model training with early stopping and metrics logging."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        module_id: str
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.module_id = module_id
        
        self.epochs = config.get("epochs", 100)
        self.patience = config.get("early_stopping_patience", 10)
        
        # Loss function
        loss_str = config.get("loss", "huber").lower()
        if loss_str == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()
            
        # Optimizer
        lr = config.get("lr", 0.001)
        weight_decay = config.get("weight_decay", 0.0001)
        opt_str = config.get("optimizer", "adamw").lower()
        if opt_str == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
    def train(self, target_checkpoint_path: str) -> Tuple[Dict[str, list], int]:
        """Trains the model and saves best checkpoint exactly at target_checkpoint_path."""
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        
        history = defaultdict(list)
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._val_epoch()
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                os.makedirs(os.path.dirname(target_checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), target_checkpoint_path)
            else:
                patience_counter += 1
                
            lr_current = self.optimizer.param_groups[0]['lr']
            best_str = "yes" if is_best else "no"
            
            print(f"[TRAIN] module={self.module_id} | epoch={epoch}/{self.epochs} | "
                  f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr={lr_current:.6f} | best={best_str}")
            
            if patience_counter >= self.patience:
                print(f"[EARLY_STOP] module={self.module_id} | stopped_epoch={epoch} | "
                      f"best_epoch={best_epoch} | best_val_loss={best_val_loss:.5f}")
                break
                
        print(f"[SAVE] module={self.module_id} | checkpoint={target_checkpoint_path}")
                
        # Load best weights before returning
        if os.path.exists(target_checkpoint_path):
            self.model.load_state_dict(torch.load(target_checkpoint_path, map_location=self.device, weights_only=True))
            
        return dict(history), best_epoch
        
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
                
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            ypred = self.model(x)
            loss = self.criterion(ypred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            
        return total_loss / len(self.train_loader.dataset)
        
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                    
                x, y = x.to(self.device), y.to(self.device)
                
                ypred = self.model(x)
                loss = self.criterion(ypred, y)
                
                total_loss += loss.item() * x.size(0)
                
        return total_loss / len(self.val_loader.dataset)
