import os
import random
import numpy as np

from glob import glob

import torch
import torch.nn as nn

import pytorch_lightning as pl

from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

#-----------------------------

NUM_RE = "[0-9]"
AUDIO_RE = "[fw][la][av]*"

MEL_WNDW_LEN = 25  
MEL_WNDW_STP = 10    
MEL_N_CHANNELS = 40 

PAR_N_FRAMES = 160

SPKR_PER_BATCH = 64
UTTER_PER_SPKR = 10

NUM_WRKR = os.cpu_count()

LRATE = 1e-4
HL_SIZE = 256
EM_SIZE = 256
NUM_LAYERS = 3

#---------------------------------------------------------
class RandomCycler:
    def __init__(self, source):
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count):
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out

    def __next__(self):
        return self.sample(1)[0]

class Utterance:
    def __init__(self, frames_fpath, ):
        self.frames_fpath = frames_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        frames = self.get_frames()
        if frames.shape[0] == n_frames: start = 0
        else: start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)

class Speaker:
    def __init__(self, root):
        self.root = root
        self.utterances = None
        self.utterance_cycler = None
        
    def load_utterances(self): 
        self.utterances = [Utterance(upath) for upath in glob(self.root + "/*")]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        if self.utterances is None: self.load_utterances()

        utterances = self.utterance_cycler.sample(count)

        return [(u,) + u.random_partial(n_frames) for u in utterances]

class SpeakerBatch:
    def __init__(self, speakers, utterances_per_speaker, n_frames):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

class SpeakerVerificationDataset(Dataset):
    def __init__(self, dataset_root):
        self.root = dataset_root
        speaker_dirs = [f for f in glob(self.root + f"/*{NUM_RE}")]
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)

class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, n_frames=PAR_N_FRAMES)

#---------------------------------------------------------------------------------------------

class SpeakerVerificationDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir=None):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir

    def train_dataloader(self):
        return SpeakerVerificationDataLoader(
            SpeakerVerificationDataset(self.train_dir),
            speakers_per_batch=SPKR_PER_BATCH,
            utterances_per_speaker=UTTER_PER_SPKR,
            num_workers=NUM_WRKR,
        )
    
    def val_dataloader(self):
        return SpeakerVerificationDataLoader(
            SpeakerVerificationDataset(self.val_dir),
            speakers_per_batch=SPKR_PER_BATCH,
            utterances_per_speaker=UTTER_PER_SPKR,
            num_workers=NUM_WRKR,
        )

# ----------------------------------------------------------------------------------
class SpeakerEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()    
        # Network defition
        self.lstm = nn.LSTM(input_size=MEL_N_CHANNELS,
                            hidden_size=HL_SIZE, 
                            num_layers=NUM_LAYERS, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=HL_SIZE, 
                                out_features=EM_SIZE)
        self.relu = torch.nn.ReLU()
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, utterances, hidden_init=None):
        _, (hidden, _) = self.lstm(utterances, hidden_init)
        
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int32)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
        
    def loss(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long()
        loss = self.loss_fn(sim_matrix, target)
        
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int32)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LRATE)

    def common_step(self, batch, batch_idx, stage=None):
        inputs = torch.from_numpy(batch.data)

        embeds = self(inputs)
        embeds_loss = embeds.view((SPKR_PER_BATCH, UTTER_PER_SPKR, -1))

        loss, eer = self.loss(embeds_loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self,  batch, batch_idx):
        return self.common_step(batch, batch_idx, "val")


class SpeakerEncoderCallbacks(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        # Gradient scale
        pl_module.similarity_weight.grad *= 0.01
        pl_module.similarity_bias.grad *= 0.01          
        # Gradient clipping
        clip_grad_norm_(pl_module.parameters(), 3, norm_type=2)



if __name__=="__main__":

    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    tdir = "../data/PreProcessed/Audio/LibriSpeech/train-clean-100"
    vdir = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    datamodule = SpeakerVerificationDataModule(tdir, vdir)

    # print(datamodule)

    model = SpeakerEncoder()
    
    model_cb = SpeakerEncoderCallbacks()

    trainer = pl.Trainer(callbacks=[model_cb],
                         fast_dev_run=False, 
                         max_epochs=5,
                         gpus=None,
                         tpu_cores=None,
                         flush_logs_every_n_steps=100,
                         log_every_n_steps=5,
                         limit_train_batches=5,
                         limit_val_batches=5,
                        )

    trainer.fit(model, datamodule=datamodule)



    ## --- UTTERANCE --- 
    # root = "../data/Preprocessed/Audio/LibriSpeech/dev-clean/1272"

    # for upath in glob(root + "/*")[:5]:
    #     utter = Utterance(upath)
    #     print(utter)
    #     print(utter.frames_fpath)
    #     print(utter.get_frames().shape)
    #     arr = utter.random_partial(PAR_N_FRAMES)
    #     print(arr[0].shape, arr[1])
    #     print("---------------")

    ## --- SPEAKER ---
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # for spath in glob(root + "/*")[:5]: 
    #     spkr = Speaker(spath)
    #     print(spkr)
    #     for _, arr, _ in spkr.random_partial(4, PAR_N_FRAMES):
    #         print(arr.shape)

    ## SPEAKER BATCH
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # speakers = [Speaker(spath) for spath in glob(root+"/*")][:3]

    # test = SpeakerBatch(speakers, 4, PAR_N_FRAMES)

    # print(test)
    # print(test.data.shape)

    ## SPEAKER_VERIFICATION_DATASET
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # test = SpeakerVerificationDataset(root)

    # print(test)
    # print(test.__getitem__(1))
    # print(test.__getitem__(1).random_partial(4, 160))

    ## SPEAKER_VERIFICATION_DATALOADER
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # test = SpeakerVerificationDataLoader(SpeakerVerificationDataset(root), 4, 5)

    # print(test)
    # print(test.dataset)
    # print(test.dataset.__getitem__(1))
    # print(test.dataset.__getitem__(1).random_partial(4, 160))

    ## SPEAKER_VERIFICATION_DATAMODULE
    # root = "../data/PreProcessed/Audio/LibriSpeech/dev-clean"

    # test = SpeakerVerificationDataModule(root)

    # print(test) 
    # print(test.train_dataloader())
    # print(test.train_dataloader().dataset)
    # print(test.train_dataloader().dataset.__getitem__(5))
    # print(test.train_dataloader().dataset.__getitem__(5).random_partial(4, 160))







    



