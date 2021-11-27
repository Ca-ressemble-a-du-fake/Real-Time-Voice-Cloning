import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from synthesizer import audio
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from vocoder.display import *
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import time
import platform
import pytorch_lightning as pl



def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")




def train(run_id: str, syn_dir: str, models_dir: str, save_every: int,
         backup_every: int, force_restart:bool, hparams):

    syn_dir = Path(syn_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)
    
    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")
    metadata_fpath = syn_dir.joinpath("train.txt")
    
    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: Tacotron")
  
    
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    
    

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size,
                     syn_dir=syn_dir,
                     hparams=hparams,
                     plot_dir=plot_dir, 
                     mel_output_dir=mel_output_dir, 
                     wav_dir=wav_dir
                     )
                     


    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Tacotron from scratch\n")
        model.save(weights_fpath)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write("{}\n".format(symbol))

    else:
        print("\nLoading weights at %s" % weights_fpath)
        #model.load(weights_fpath, optimizer)
        model.load(weights_fpath)
        print("Tacotron weights loaded from step %d" % model.step)
    



    # Lightning additions
    trainer = pl.Trainer(default_root_dir=model_dir, 
    gpus=1, 
    precision=16,
    #fast_dev_run=True, 
    #log_every_n_steps=1
                        )
    
    trainer.fit(model)


