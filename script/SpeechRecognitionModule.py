import os
import ast
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from script.Data import Data, collate_fn_padd
from script.SpeechRecognition import SpeechRecognition
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class SpeechRecognitionModule(LightningModule):

    def __init__(self, model, args):
        super(SpeechRecognitionModule, self).__init__()
        self.model = model
        self.loss_function = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
        return [optimizer], [scheduler]

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch 
        batch_size = spectrograms.shape[0]
        hidden = self.model._init_hidden(batch_size)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = nn.functional.log_softmax(output, dim=2)
        loss = self.loss_function(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizers().param_groups[0]['lr']}
        return {'loss': loss, 'log': logs}

    def train_dataloader(self):
        data_params = Data.parameters
        data_params.update(self.args.dparams_override)
        train_dataset = Data(json_path=self.args.train_file, **data_params)
        return DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, num_workers=self.args.data_workers,
                          pin_memory=True, collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.lr_schedulers().step(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        data_params = Data.parameters
        data_params.update(self.args.dparams_override)
        valid_dataset = Data(json_path=self.args.valid_file, **data_params, valid=True)
        return DataLoader(dataset=valid_dataset, batch_size=self.args.batch_size, num_workers=self.args.data_workers,
                          pin_memory=True, collate_fn=collate_fn_padd)


def create_checkpoint_callback(args):
    return ModelCheckpoint(
        dirpath=args.save_model_path,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename='{epoch}-{val_loss:.2f}'
    )


def main(args):
    hparams = SpeechRecognition.hyper_parameters
    hparams.update(args.hparams_override)
    model = SpeechRecognition(**hparams)

    if args.load_model_from:
        speech_module = SpeechRecognitionModule.load_from_checkpoint(args.load_model_from, model=model, args=args)
    else:
        speech_module = SpeechRecognitionModule(model, args)

    logger = TensorBoardLogger(save_dir=args.logdir, name='speech_recognition')
    trainer = Trainer(
        max_epochs=args.epochs, gpus=args.gpus, num_nodes=args.nodes, strategy=None,
        logger=logger, gradient_clip_val=1.0, val_check_interval=args.valid_every,
        callbacks=[create_checkpoint_callback(args)], resume_from_checkpoint=args.resume_from_checkpoint
    )
    trainer.fit(speech_module)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for speech recognition model.")
    parser.add_argument('-n', '--nodes', default=1, type=int, help='Number of nodes for training.')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of GPUs per node.')
    parser.add_argument('-w', '--data_workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str, help='Distributed backend to use.')

    parser.add_argument('--train_file', required=True, type=str, help='Path to training data JSON file.')
    parser.add_argument('--valid_file', required=True, type=str, help='Path to validation data JSON file.')
    parser.add_argument('--valid_every', default=1000, type=int, help='Validate after every N iterations.')

    parser.add_argument('--save_model_path', required=True, type=str, help='Directory to save the model.')
    parser.add_argument('--load_model_from', type=str, help='Path to a pretrained model checkpoint.')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to resume from a checkpoint.')
    parser.add_argument('--logdir', default='tb_logs', type=str, help='Directory to save logs.')

    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training.')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate.')
    parser.add_argument('--pct_start', default=0.3, type=float, help='Percentage of growth phase in one cycle.')
    parser.add_argument('--div_factor', default=100, type=int, help='Div factor for one cycle.')
    parser.add_argument('--hparams_override', default="{}", type=str, help='Override hyperparameters as a dictionary.')
    parser.add_argument('--dparams_override', default="{}", type=str, help='Override data parameters as a dictionary.')

    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)

    if args.save_model_path and not os.path.isdir(os.path.dirname(args.save_model_path)):
        raise Exception(f"The directory for path {args.save_model_path} does not exist.")

    main(args)