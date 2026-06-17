import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import librosa
import soundfile as sf
import argparse
import shutil


@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, n_fft=400, hop=100, save_tracks=False
):
    name = os.path.split(audio_path)[-1]
    noisy, sr = librosa.load(audio_path,sr=16000)
    c = np.sqrt(len(noisy) / np.sum((noisy**2.0)))
    noisy = noisy*c
    num_frame = np.ceil(len(noisy)/(2*sr))
    padded_len = int(2*sr*num_frame-len(noisy))
    noisy = np.pad(noisy,padded_len)
    output = np.zeros(len(noisy))
    seg_noise = librosa.util.frame(noisy,frame_length=2*sr,hop_length=2*sr)
    for cnt in range(seg_noise.shape[1]):

        input_wave = torch.from_numpy(seg_noise[:,cnt]).cuda().unsqueeze(0)        

        noisy_spec = torch.view_as_real(torch.stft(
            input_wave, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True,return_complex=True
        ))
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        est_real, est_imag = model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_uncompress = torch.complex(est_spec_uncompress[...,0], est_spec_uncompress[...,1])
        est_audio = torch.istft(
            est_spec_uncompress,
            n_fft,
            hop,
            window=torch.hamming_window(n_fft).cuda(),
            onesided=True,
        )
        output[cnt*2*sr:(cnt+1)*2*sr] = torch.flatten(est_audio).detach().cpu().numpy()
    output = output / c
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, output, sr)

    return est_audio


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    torch.cuda.empty_cache()
    n_fft = 400
    model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1).cuda()
    model.load_state_dict((torch.load(model_path)))
    model.eval()

    # if os.path.exists(saved_dir):
    #     shutil.rmtree(saved_dir)
    # os.mkdir(saved_dir)
        
    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    num = len(audio_list)
    count_num = 0 
    # metrics_total = np.zeros(6)
    for audio in audio_list:
        if audio.endswith('.wav'):
            count_num +=1
            if os.path.exists(os.path.join(saved_dir,audio)):
                continue
            noisy_path = os.path.join(noisy_dir, audio)
            # clean_path = os.path.join(clean_dir, audio.split('_')[-1])
            est_audio = enhance_one_track(
                model, noisy_path, saved_dir, n_fft, n_fft // 4, save_tracks
            )
            print('{:.2f}%'.format(count_num/num*100))
    #         clean_audio, sr = sf.read(clean_path)
    #         assert sr == 16000
    #         metrics = compute_metrics(clean_audio, est_audio, sr, 0)
    #         metrics = np.array(metrics)
    #         metrics_total += metrics

    # metrics_avg = metrics_total / num
    # print(
    #     "pesq: ",
    #     metrics_avg[0],
    #     "csig: ",
    #     metrics_avg[1],
    #     "cbak: ",
    #     metrics_avg[2],
    #     "covl: ",
    #     metrics_avg[3],
    #     "ssnr: ",
    #     metrics_avg[4],
    #     "stoi: ",
    #     metrics_avg[5],
    # )


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='./best_ckpt/ckpt_80',
                    help="the path where the model is saved")
parser.add_argument("--noisy_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--clean_dir", type=str, default='dir to your VCTK-DEMAND test dataset',
                    help="clean tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

args = parser.parse_args()


if __name__ == "__main__":
    noisy_dir = args.noisy_dir
    clean_dir = args.clean_dir
    evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir)
