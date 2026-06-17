import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import librosa
import torchaudio
import soundfile as sf
import argparse
import torch


@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=400, hop=160, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = librosa.load(audio_path,sr=16000)
    chunck_size = cut_len
    # noisy = noisy/np.max(noisy)
    # assert sr == 16000
    if len(noisy)%chunck_size!=0:
        noisy = np.pad(noisy,[0,chunck_size-len(noisy)%chunck_size])
    waveforms = []
    for i in range(len(noisy)//chunck_size):
        waveforms.append(noisy[i*chunck_size:(i+1)*chunck_size])
    output = np.zeros(1)
    for input_wave in waveforms:
        noisy = torch.from_numpy(input_wave).cuda().unsqueeze(0)

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)

    # length = noisy.size(-1)
    # frame_num = int(np.ceil(length / 100))
    # padded_len = frame_num * 100
    # padding_len = padded_len - length
    # noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    # if padded_len > cut_len:
    #     batch_size = int(np.ceil(padded_len / cut_len))
    #     while 100 % batch_size != 0:
    #         batch_size += 1
    #     noisy = torch.reshape(noisy, (batch_size, -1))

        noisy_spec = torch.view_as_real(torch.stft(
            noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True, return_complex=True
        ))
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag = model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_uncompress=torch.complex(est_spec_uncompress[...,0], est_spec_uncompress[...,1])
        est_audio = torch.istft(
            est_spec_uncompress,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).cuda(),
            onesided=True,
        )
        est_audio = est_audio / c
        est_audio = torch.flatten(est_audio).cpu().numpy()
        output = np.concatenate((output,est_audio))

    # assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, output, sr)

    return est_audio



def evaluation(model_path, noisy_dir, save_tracks, saved_dir):
    n_fft = 400
    model = generator.TSCNet(num_channel=128, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio = args.test_dir
    # for audio in audio_list:
        # noisy_path = os.path.join(noisy_dir, audio)
    est_audio = enhance_one_track(
        model, audio, saved_dir, 1600 * 5, n_fft, 160, save_tracks
    )


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, 
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args = parser.parse_args()


if __name__ == "__main__":
    noisy_dir = os.path.join(args.test_dir, "noisy")
    evaluation(args.model_path, noisy_dir, args.save_tracks, args.save_dir)