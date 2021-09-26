#copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import librosa
import torchaudio
import random
from argparse import ArgumentParser
import pdb

from gwdataset import GW_Dataset, WaveformDatasetTorch
from torch.utils.tensorboard import SummaryWriter
from params import AttrDict, params as base_params
from model import DiffWave

from os import path
from glob import glob
from tqdm import tqdm

random.seed(23)

models = {}


def load_model(model_dir=None, args=None, params=None, device=torch.device('cuda')):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DiffWave(args, AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
  model = models[model_dir]
  model.params.override(params)
      
  return model
      
def inference_schedule(model, fast_sampling=False):
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    print("alpha_cum",talpha_cum)
    print("gamma_cum",alpha_cum)
    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1): 
      sigmas[n] = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
    print("sigmas",sigmas)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)
    return alpha, beta, alpha_cum,sigmas, T
      

#def predict(spectrogram, model, noisy_signal, alpha, beta, alpha_cum, sigmas, T, device=torch.device('cuda')):
#  with torch.no_grad():
#    # Expand rank 2 tensors by adding a batch dimension.
#    if len(spectrogram.shape) == 2:
#      spectrogram = spectrogram.unsqueeze(0)
#    spectrogram = spectrogram.to(device)
    
    
    # audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
#    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    # pdb.set_trace()
#    noisy_audio = torch.zeros(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
#    noisy_audio[:,:noisy_signal.shape[0]] = torch.from_numpy(noisy_signal).to(device)
    
#    audio = noisy_audio
#    gamma = [0 for i in alpha] # the first 2 num didn't use
#    for n in range(len(alpha)):
#      gamma[n] = sigmas[n] 
#    gamma[0] = 0.2
#    #print("gamma",gamma)
#    for n in range(len(alpha) - 1, -1, -1):
#      c1 = 1 / alpha[n]**0.5
#      c2 = beta[n] / (1 - alpha_cum[n])**0.5
#      predicted_noise =  model(audio, spectrogram, torch.tensor([T[n]], device=audio.device)).squeeze(1)
#      mu = audio - c2 * predicted_noise
#      audio = c1 * ((1-gamma[n])*mu+gamma[n]*noisy_audio)
#      if n > 0:
#        noise = torch.randn_like(audio)
#        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
#        newsigma= max(0,sigma - c1 * gamma[n])
#        audio += newsigma * noise
#      audio = torch.clamp(audio, -1.0, 1.0)
#  return audio, model.params.sample_rate


def predict(spectrogram, model, noisy_signal, alpha, beta, alpha_cum, sigmas, T, device=torch.device('cuda')):
  with torch.no_grad():
    # Expand rank 2 tensors by adding a batch dimension.
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)
    
    
    # audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    # pdb.set_trace()
    noisy_audio = torch.zeros(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noisy_audio[:,:noisy_signal.shape[0]] = torch.from_numpy(noisy_signal).to(device)
    
    audio = noisy_audio
    gamma = [0 for i in alpha] # the first 2 num didn't use
    for n in range(len(alpha)):
      gamma[n] = sigmas[n] 
    gamma[0] = 0.2
    #print("gamma",gamma)
    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      predicted_noise =  model(audio, spectrogram, torch.tensor([T[n]], device=audio.device)).squeeze(1)
      mu = audio - c2 * predicted_noise
      audio = c1 * ((1-gamma[n])*mu+gamma[n]*noisy_audio)
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        newsigma= max(0,sigma - c1 * gamma[n])
        audio += newsigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate



def main(args):
  if args.se:
    base_params.n_mels = 513
  else:
    base_params.n_mels = 80
  #specnames = []
  #print("spectrum:",args.spectrogram_path)
  #print("noisy_signal:",args.wav_path)
  #for path in args.spectrogram_path:
  #  specnames += glob(f'{path}/*.wav.spec.npy', recursive=True)
  Dataset = GW_Dataset()
  Dataset.psd = Dataset.get_psd_2()
  Dataset.load_waveform(DIR=args.dataset_dir,data_fn=args.data_fn)
  model = load_model(model_dir=args.model_dir ,args=args)
  #alpha, beta, alpha_cum, T = inference_schedule(model, fast_sampling=args.fast)

  alpha, beta, alpha_cum, sigmas, T = inference_schedule(model, fast_sampling=args.fast)
  idx=500
  clean = Dataset.test_waveform['clean'][idx]
  noisy = Dataset.change_data_snr(clean,200)
  spec, _, _ = Dataset.make_spectrum(y=noisy,FRAMELENGTH=base_params.n_fft, SHIFT=base_params.hop_samples)  
  spectrogram = torch.from_numpy(spec).type(torch.FloatTensor)
  noisy_signal = noisy
  wlen = noisy_signal.shape[0]
  audio, sr = predict(spectrogram, model, noisy_signal, alpha, beta, alpha_cum, sigmas, T)
  audio = audio[:,:wlen]
  data = audio.to('cpu').numpy()[0]
  fn = os.path.join(args.output,'denoisied_waveform.npy')
  np.save(fn,data)



if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('dataset_dir',
      help='space separated list of directories from spectrogram file generated by diffwave.preprocess')
  parser.add_argument('data_fn',
      help='input noisy wav directory')
  parser.add_argument('--output', '-o', default='output/',
      help='output path name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  parser.add_argument('--se', dest='se', action='store_true')
  parser.add_argument('--vocoder', dest='se', action='store_false')
  parser.add_argument('--voicebank', dest='voicebank', action='store_true')
  parser.set_defaults(se=True)
  parser.set_defaults(fix_in=False)
  parser.set_defaults(voicebank=False)
  main(parser.parse_args())
