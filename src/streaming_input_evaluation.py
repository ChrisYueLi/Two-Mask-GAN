import torch
import time
import numpy as np
import numpy as np
import os
from utils import *
import librosa
import soundfile as sf
import argparse
import torch
import librosa
from runtime import build_generator, load_generator_checkpoint, resolve_device, str2bool






INDEX=-3


@torch.no_grad()
def enhance_one_track(
    model, audio_path, saved_dir, device, n_fft=400, hop=160, save_tracks=False
):
    name = os.path.split(audio_path)[-1][:-4]+'_streaming.wav'
    noisy, sr = librosa.load(audio_path,sr=16000)
    noisy = noisy/np.max(noisy)
    buffer_len = 500
    chunck_size = buffer_len*16
    num_buffer = 2000//buffer_len
    
    if len(noisy)%chunck_size!=0:
        noisy = np.pad(noisy,[0,4*chunck_size-len(noisy)%chunck_size])
    waveforms = []
    # Cut the audio into small audio chunks
    for i in range(len(noisy)//chunck_size):
        waveforms.append(noisy[i*chunck_size:(i+1)*chunck_size])
    print(len(waveforms))
    print(len(waveforms[0]))
    output = np.zeros(len(noisy))
    mean_time = 0
    input_mag = np.zeros((1,201,201))
    mask_add = np.zeros((1,201,201))
    mask_mul = np.zeros((1,201,201))
    output_mag = np.zeros((1,201,201))
    for cnt in range(len(waveforms)):
        temp_cnt = cnt
        input_wave = None
        # Make sure every chunck has the same length
        if len(waveforms[cnt])!=chunck_size:
            print("ERROR LENGTH, PADDING")
            waveforms[cnt] = np.pad(waveforms[cnt],[0,chunck_size-len(waveforms[cnt])])
        input_wave = waveforms[cnt]
        temp_cnt-=1
        # Concatenate the current buffer with previous buffer.
        while temp_cnt>=0 and len(input_wave)<num_buffer*chunck_size:
            input_wave = np.concatenate((waveforms[temp_cnt],input_wave))
            temp_cnt-=1
        # Check whether the length matches with the pre-set length.
        if len(input_wave)<num_buffer*chunck_size:
            input_wave = np.pad(input_wave,[num_buffer*chunck_size-len(input_wave),0])

        if input_wave is not None:
            stime = time.time()
            input_wave = torch.from_numpy(input_wave).to(device).unsqueeze(0)
            c = torch.sqrt(input_wave.size(-1) / torch.sum((input_wave**2.0), dim=-1))
            input_wave = torch.transpose(input_wave, 0, 1)
            input_wave = torch.transpose(input_wave * c, 0, 1)
            noisy_spec = torch.view_as_real(torch.stft(
                input_wave, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True, return_complex=True
            ))
            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
            
            # est_real, est_imag, temp_mask_add, temp_mask_mul = model(noisy_spec)

            # est_real, est_imag, temp_mask_mul = model(noisy_spec)
            
            est_real, est_imag = model(noisy_spec)
            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
            
            # mask_add, mask_mul = np.concatenate((mask_add,temp_mask_add.permute(0, 1, 3, 2).squeeze(0).detach().cpu().numpy()),axis=0), np.concatenate((mask_mul,temp_mask_mul.permute(0, 1, 3, 2).squeeze(0).detach().cpu().numpy()),axis=0)
            
            # mask_mul = np.concatenate((mask_mul,temp_mask_mul.permute(0, 1, 3, 2).squeeze(0).detach().cpu().numpy()),axis=0)

            temp_mag =torch.sqrt(noisy_spec.permute(0, 1, 3, 2)[:, 0, :, :] ** 2 + noisy_spec.permute(0, 1, 3, 2)[:, 1, :, :] ** 2).unsqueeze(1)
            input_mag = np.concatenate((input_mag,temp_mag.squeeze(0).detach().cpu().numpy()),axis=0)
            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_spec_uncompress=torch.complex(est_spec_uncompress[...,0], est_spec_uncompress[...,1])
            output_mag = np.concatenate((output_mag,torch.abs(est_spec_uncompress).detach().cpu().numpy()),axis=0)
            est_audio = torch.istft(
                est_spec_uncompress,
                n_fft,
                hop,
                window=torch.hamming_window(n_fft).to(device),
                onesided=True,
            )
            est_audio = est_audio / c
            est_audio = torch.flatten(est_audio).detach().cpu().numpy()
            mean_time+=time.time()-stime
            if cnt>=2:
                output[(cnt-2)*chunck_size:(cnt)*chunck_size] += est_audio[-chunck_size*2:]
            elif cnt<2:
                output[:(cnt+1)*chunck_size] += est_audio[-chunck_size*(cnt+1):]
            # else:
            #     output[(cnt-1)*chunck_size:] += est_audio[-chunck_size*2:]
            
            # assert len(est_audio) == length
    print(len(mask_mul))
    print("Average Processing time is {:.3f}s".format(mean_time/len(waveforms)))
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, output/np.max(output), sr)
    # target_path = audio_path.replace('result','target')

    # target_path = audio_path[:-10]+'target.wav'
    # target_sig,fs = librosa.load(target_path,sr=16000)
    # target_sig = librosa.util.frame(target_sig,frame_length=fs*2,hop_length=fs//2)[:,INDEX+1]
    # target_tf = librosa.stft(target_sig,n_fft=400,win_length=400,hop_length=160)
    # fig,ax = plt.subplots(nrows=5,ncols=1,sharex=True)
    # imp1=display.specshow(mask_add[-1,:,:],y_axis='log',ax=ax[0])
    # fig.colorbar(imp1)
    # ax[0].set_title('Mask ADD',y=1.0,pad=-0.2)
    # imp2=display.specshow(mask_mul[INDEX,:,:],y_axis='log',ax=ax[1])
    # ax[1].set_title('Mask MUL',y=1.0,pad=-0.2)
    # fig.colorbar(imp2)
    # imp3=display.specshow(input_mag[INDEX,:,:],y_axis='log',ax=ax[2])
    # ax[2].set_title('INPUT',y=1.0,pad=-0.2)
    # fig.colorbar(imp3)
    # imp4=display.specshow(output_mag[INDEX,:,:],y_axis='log',ax=ax[3])
    # ax[3].set_title('OUTPUT',y=1.0,pad=-0.2)
    # fig.colorbar(imp4)
    # # imp5=display.specshow(np.abs(target_tf),y_axis='log',ax=ax[4])
    # # ax[4].set_title('Target',y=1.0,pad=-0.2)
    # # fig.colorbar(imp5)
    # fig.show()
    # plt.savefig(os.path.join(saved_dir, f"{name}.png"))


    return est_audio



def evaluation(model_path, noisy_dir, save_tracks, saved_dir, device, num_channel, mask_mode, module):
    n_fft = 400
    model = build_generator(device, n_fft=n_fft, num_channel=num_channel, mask_mode=mask_mode, module=module)
    load_generator_checkpoint(model, model_path, device)
    model.eval()

    os.makedirs(saved_dir, exist_ok=True)

    audio = args.test_dir
    # for audio in audio_list:
        # noisy_path = os.path.join(noisy_dir, audio)
    est_audio = enhance_one_track(
        model, audio, saved_dir, device, n_fft, 160, save_tracks
    )




parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,
                    help="the path where the model is saved")
parser.add_argument("--test_dir", type=str, required=True,
                    help="noisy tracks dir to be enhanced")
parser.add_argument("--save_tracks", type=str2bool, default=True, help="save predicted tracks or not")
parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")
parser.add_argument("--device", type=str, default="cuda", help="torch device")
parser.add_argument("--num_channel", type=int, default=128, help="TSCNet channel width")
parser.add_argument("--mask_mode", type=str, default="add", choices=["add", "mul"], help="masking mode")
parser.add_argument("--module", type=str, default="conformer", choices=["conformer", "mamba"], help="sequence module")

args = parser.parse_args()


if __name__ == "__main__":
    device = resolve_device(args.device)
    noisy_dir = os.path.join(args.test_dir, "noisy")
    evaluation(
        args.model_path,
        noisy_dir,
        args.save_tracks,
        args.save_dir,
        device,
        args.num_channel,
        args.mask_mode,
        args.module,
    )
