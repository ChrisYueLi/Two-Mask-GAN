import torch
import torchaudio
import time
from torchaudio.io import StreamReader,StreamWriter
import numpy as np
import numpy as np
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import librosa
import torchaudio
import soundfile as sf
import argparse
import torch
import librosa
from librosa import display
import matplotlib.pyplot as plt
from runtime import build_generator, load_generator_checkpoint, resolve_device, str2bool









@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, device, n_fft=400, hop=160, save_tracks=False, streaming=1
):
    
    noisy, sr = librosa.load(audio_path,sr=16000)
    # noisy = noisy/np.max(noisy)
    if bool(streaming):
        name = os.path.split(audio_path)[-1][:-4]+'_streaming.wav'
        stime = time.time()
        buffer_len = 500
        chunck_size = buffer_len*16
        num_buffer = 2000//buffer_len
        if len(noisy)%chunck_size!=0:
            noisy = np.pad(noisy,[3*chunck_size,4*chunck_size-len(noisy)%chunck_size])
        waveforms = []
        for i in range(len(noisy)//chunck_size):
            waveforms.append(noisy[i*chunck_size:(i+1)*chunck_size])
        output = np.zeros(len(noisy)-3*chunck_size)
        for cnt in range(3,len(waveforms)-3):
            temp_cnt = cnt
            input_wave = None
            # Make sure every chunck has the same length
            if len(waveforms[cnt])!=chunck_size:
                print(cnt)
                waveforms[cnt] = np.pad(waveforms[cnt],[0,chunck_size-len(waveforms[cnt])])
            input_wave = waveforms[cnt]
            temp_cnt-=1
            while temp_cnt>=0 and len(input_wave)<num_buffer*chunck_size:
                input_wave = np.concatenate((waveforms[temp_cnt],input_wave))
                temp_cnt-=1
            assert len(input_wave) == num_buffer*chunck_size,print(len(input_wave))
            if input_wave is not None:
                input_wave = torch.from_numpy(input_wave).to(device).unsqueeze(0)

                c = torch.sqrt(input_wave.size(-1) / torch.sum((input_wave**2.0), dim=-1))
                input_wave = torch.transpose(input_wave, 0, 1)
                input_wave = torch.transpose(input_wave * c, 0, 1)
                noisy_spec = torch.view_as_real(torch.stft(
                    input_wave, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True, return_complex=True
                ))
                noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
                est_real, est_imag= model(noisy_spec)
                est_real, est_imag= model(noisy_spec)
                est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
                est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
                est_spec_uncompress=torch.complex(est_spec_uncompress[...,0], est_spec_uncompress[...,1])
                est_audio = torch.istft(
                    est_spec_uncompress,
                    n_fft,
                    hop,
                    window=torch.hamming_window(n_fft).to(device),
                    onesided=True,
                )
                est_audio = est_audio / c
                est_audio = torch.flatten(est_audio).detach().cpu().numpy()
                if num_buffer!=1:
                    if cnt>=2 and cnt<len(waveforms)-1:
                        output[(cnt-2)*chunck_size:cnt*chunck_size] += est_audio[-chunck_size*2:]
                else:
                    output[cnt*chunck_size:(cnt+1)*chunck_size]+=est_audio
    else:
        name = os.path.split(audio_path)[-1][:-4]+'_whole.wav'
        stime = time.time()
        input_wave = torch.from_numpy(noisy).to(device).unsqueeze(0)
        c = torch.sqrt(input_wave.size(-1) / torch.sum((input_wave**2.0), dim=-1))
        input_wave = torch.transpose(input_wave, 0, 1)
        input_wave = torch.transpose(input_wave * c, 0, 1)
        noisy_spec = torch.view_as_real(torch.stft(
            input_wave, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True, return_complex=True
        ))
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag= model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_uncompress=torch.complex(est_spec_uncompress[...,0], est_spec_uncompress[...,1])
        est_audio = torch.istft(
            est_spec_uncompress,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).to(device),
            onesided=True,
        )

        input_wave = torch.from_numpy(noisy).to(device).unsqueeze(0)
        est_audio = est_audio / c
        output = torch.flatten(est_audio).detach().cpu().numpy()
        print("Processing Time is {:.2f}".format(time.time()-stime))


    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, output, sr)

    return est_audio



def evaluation(model_path, noisy_dir, save_tracks, saved_dir, streaming, device, num_channel, mask_mode, module):
    n_fft = 400
    model = build_generator(device, n_fft=n_fft, num_channel=num_channel, mask_mode=mask_mode, module=module)
    load_generator_checkpoint(model, model_path, device)
    model.eval()

    os.makedirs(saved_dir, exist_ok=True)

    audio = args.test_dir
    # for audio in audio_list:
        # noisy_path = os.path.join(noisy_dir, audio)
    est_audio = enhance_one_track(
        model, audio, saved_dir, device, n_fft, 160, save_tracks,streaming
    )




parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, required=True,
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str2bool, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")
parser.add_argument("--streaming", type=int, default=1, help="whether adopting streaming method")
parser.add_argument("--device", type=str, default="cuda", help="torch device")
parser.add_argument("--num_channel", type=int, default=128, help="TSCNet channel width")
parser.add_argument("--mask_mode", type=str, default="add", choices=["add", "mul"], help="masking mode")
parser.add_argument("--module", type=str, default="conformer", choices=["conformer", "mamba"], help="sequence module")

args = parser.parse_args()


if __name__ == "__main__":
    device = resolve_device(args.device)
    # noisy_dir = os.path.join(args.test_dir, "noisy")
    noisy_dir = args.test_dir
    evaluation(
        args.model_path,
        noisy_dir,
        args.save_tracks,
        args.save_dir,
        args.streaming,
        device,
        args.num_channel,
        args.mask_mode,
        args.module,
    )
