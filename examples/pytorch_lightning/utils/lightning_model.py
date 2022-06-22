import logging
from argparse import Namespace
from typing import Union

import torch
from pytorch_lightning import LightningModule
from torch.nn import Module, CrossEntropyLoss
from torchmetrics import Accuracy, F1Score, Precision, Recall

from .unused_args import clean_hyperparameters
from .utils import create_optimizer, create_scheduler


class ModelWrapper(LightningModule):
    def __init__(
        self,
        model: Module,
        num_classes: int,
        args: Namespace,
        add_f1_prec_recall: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(clean_hyperparameters(args))
        self.loss_function = CrossEntropyLoss()
        self.model = model
        self.train_accuracy_top1 = Accuracy(num_classes=num_classes)
        self.train_accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.accuracy_top1 = Accuracy(num_classes=num_classes)
        self.accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.add_f1_prec_recall = add_f1_prec_recall
        if add_f1_prec_recall:
            self.f1 = F1Score(num_classes=num_classes)
            self.prec = Precision(num_classes=num_classes)
            self.recall = Recall(num_classes=num_classes)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        x_train, y_train = batch

        y_hat = self.model(x_train)
        loss = self.loss_function(y_hat, y_train)
        self.train_accuracy_top1(y_hat, y_train)
        self.train_accuracy_top5(y_hat, y_train)
        self.log_dict(
            {
                "metrics/train-top1-accuracy": self.train_accuracy_top1,
                "metrics/train-top5-accuracy": self.train_accuracy_top5,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log("loss/train", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # type: ignore
        x_test, y_test = batch

        y_hat = self.model(x_test)
        loss = self.loss_function(y_hat, y_test)

        self.accuracy_top1(y_hat, y_test)
        self.accuracy_top5(y_hat, y_test)

        metrics_dict = {
            "metrics/test-top1-accuracy": self.accuracy_top1,
            "metrics/test-top5-accuracy": self.accuracy_top5,
            "loss/test": loss,
        }

        if self.add_f1_prec_recall:
            self.f1(y_hat, y_test)
            self.prec(y_hat, y_test)
            self.recall(y_hat, y_test)
            metrics_dict.update(
                {
                    "metrics/f1": self.f1,
                    "metrics/precision": self.prec,
                    "metrics/recall": self.recall,
                }
            )
        self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Union[dict, torch.optim.Optimizer]:  # type: ignore
        logging.info(f"Using {self.hparams.optimizer} optimizer and {self.hparams.lr_scheduler} lr scheduler...")
        optimizer = create_optimizer(self.hparams.optimizer, self.model, self.hparams.lr, self.hparams.momentum)
        if self.hparams.lr_scheduler is not None:
            scheduler = create_scheduler(
                self.hparams.lr_scheduler,
                optimizer,
                self.hparams.lr_factor,
                self.hparams.lr_steps,
                self.hparams.max_epochs,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
