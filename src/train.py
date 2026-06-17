from models.generator import TSCNet
from models import discriminator
import os
from data import dataloader
import torch.nn.functional as F
import torch
import torchaudio
from utils import power_compress, power_uncompress
import logging
from torchinfo import summary
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./logs", help="dir for training logs")
parser.add_argument("--log_audio_interval", type=int, default=1, help="save eval audio every N steps")
parser.add_argument("--log_audio_samples", type=int, default=20, help="num of eval samples to save")
parser.add_argument("--sample_rate", type=int, default=16000, help="audio sample rate for logs")

parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
parser.add_argument("--init_lr", type=float, default=5e-4, help="initial learning rate")
parser.add_argument("--cut_len", type=int, default=16000*2, help="cut length, default is 2 seconds in denoise "
                                                                 "and dereverberation")
parser.add_argument("--data_dir", type=str, default='dir to VCTK-DEMAND dataset',
                    help="dir of VCTK+DEMAND dataset")
parser.add_argument("--save_model_dir", type=str, default='./saved_model',
                    help="dir of saved model")
parser.add_argument("--loss_weights", type=list, default=[0.1, 0.9, 0.2, 0.05],
                    help="weights of RI components, magnitude, time loss, and Metric Disc")
parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume training")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    @staticmethod
    def peak_normalize(x, eps=1e-8):
        peak = x.abs().max()
        if peak > 1:
            return x / (peak + eps)
        return x

    def _to_mel(self, wav):
        mel = self.mel(wav)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        elif mel.dim() == 4 and mel.size(1) != 1:
            mel = mel.mean(dim=1, keepdim=True)
        return mel

    def _init_logging(self):
        self.is_main = self.gpu_id == 0
        self.log_dir = Path(args.log_dir)
        self.samples_dir = self.log_dir / "samples"
        if self.is_main:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self._test_iter = None

    def _next_test_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_ds)
        try:
            batch = next(self._test_iter)
        except StopIteration:
            self._test_iter = iter(self.test_ds)
            batch = next(self._test_iter)
        return batch

    @torch.no_grad()
    def _log_eval_step(self, epoch, step, global_step, train_loss, train_disc_loss):
        if not self.is_main:
            return
        batch = self._next_test_batch()
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.size(0)).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(clean, noisy)
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        val_loss, val_loss_mag, val_loss_ri, val_time_loss, val_gan_loss = (
            self.calculate_generator_loss(generator_outputs)
        )
        val_disc_loss = self.calculate_discriminator_loss(generator_outputs)
        if val_disc_loss is None:
            val_disc_loss = torch.tensor([0.0])

        self.writer.add_scalar("train/loss", float(train_loss), global_step)
        self.writer.add_scalar("train/disc_loss", float(train_disc_loss), global_step)
        self.writer.add_scalar("val/loss", float(val_loss.item()), global_step)
        self.writer.add_scalar("val/disc_loss", float(val_disc_loss.item()), global_step)
        self.writer.add_scalar("val/loss_mag", float(val_loss_mag.item()), global_step)
        self.writer.add_scalar("val/loss_ri", float(val_loss_ri.item()), global_step)
        self.writer.add_scalar("val/time_loss", float(val_time_loss.item()), global_step)
        self.writer.add_scalar("val/gan_loss", float(val_gan_loss.item()), global_step)

        # Audio logging disabled.


    def __init__(self, train_ds, test_ds, gpu_id: int):
        self.n_fft = 400
        self.hop = 100
        self.n_mels = 80
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            power=2.0,
        ).to(gpu_id)
        self.model = TSCNet(
            num_channel=128, num_features=self.n_fft // 2 + 1, mask_mode="add", module='conformer'
        ).cuda()
        summary(
            self.model, [(1, 2, args.cut_len // self.hop + 1, int(self.n_fft / 2) + 1)]
        )
        self.discriminator = discriminator.Discriminator(ndf=16).cuda()
        summary(
            self.discriminator,
            [
                (1, 1, self.n_mels, args.cut_len // self.hop + 1),
                (1, 1, self.n_mels, args.cut_len // self.hop + 1),
            ],
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr)
        self.optimizer_disc = torch.optim.AdamW(
            self.discriminator.parameters(), lr=2 * args.init_lr
        )

        self.model = DDP(self.model, device_ids=[gpu_id])
        self.discriminator = DDP(self.discriminator, device_ids=[gpu_id])
        self.gpu_id = gpu_id
        self.global_step = 0
        self._init_logging()

    def forward_generator_step(self, clean, noisy):

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(
            clean * c, 0, 1
        )

        noisy_spec = torch.stft(
            noisy,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
            return_complex=False,
        )
        clean_spec = torch.stft(
            clean,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
            return_complex=False,
        )
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_complex = torch.complex(
            est_spec_uncompress[..., 0], est_spec_uncompress[..., 1]
        )
        est_audio = torch.istft(
            est_spec_complex,
            self.n_fft,
            self.hop,
            window=torch.hamming_window(self.n_fft).to(self.gpu_id),
            onesided=True,
        )
        clean_mel = self._to_mel(clean)
        est_mel = self._to_mel(est_audio)

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
            "clean_mel": clean_mel,
            "est_mel": est_mel,
        }

    def calculate_generator_loss(self, generator_outputs):

        predict_fake_metric = self.discriminator(
            generator_outputs["clean_mel"], generator_outputs["est_mel"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            args.loss_weights[0] * loss_ri
            + args.loss_weights[1] * loss_mag
            + args.loss_weights[2] * time_loss
            + args.loss_weights[3] * gen_loss_GAN
        )

        return loss, loss_mag, loss_ri, time_loss, gen_loss_GAN

    def calculate_discriminator_loss(self, generator_outputs, use_pesq=True):
        predict_enhance_metric = self.discriminator(
            generator_outputs["clean_mel"], generator_outputs["est_mel"].detach()
        )
        predict_max_metric = self.discriminator(
            generator_outputs["clean_mel"], generator_outputs["clean_mel"]
        )

        if not use_pesq:
            zero_labels = torch.zeros_like(generator_outputs["one_labels"])
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten(), zero_labels)
            return discrim_loss_metric

        length = generator_outputs["est_audio"].size(-1)
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            discrim_loss_metric = F.mse_loss(
                predict_max_metric.flatten(), generator_outputs["one_labels"]
            ) + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = None

        return discrim_loss_metric

    def train_step(self, batch):

        # Trainer generator
        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.size(0)).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss, loss_mag, loss_ri, time_loss, gen_loss_GAN = self.calculate_generator_loss(
            generator_outputs
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train Discriminator
        discrim_loss_metric = self.calculate_discriminator_loss(
            generator_outputs, use_pesq=False
        )

        if discrim_loss_metric is not None:
            self.optimizer_disc.zero_grad()
            discrim_loss_metric.backward()
            self.optimizer_disc.step()
        else:
            discrim_loss_metric = torch.tensor([0.0])

        return (
            loss.item(),
            discrim_loss_metric.item(),
            loss_mag.item(),
            loss_ri.item(),
            time_loss.item(),
            gen_loss_GAN.item(),
        )

    @torch.no_grad()
    def test_step(self, batch):

        clean = batch[0].to(self.gpu_id)
        noisy = batch[1].to(self.gpu_id)
        one_labels = torch.ones(clean.size(0)).to(self.gpu_id)

        generator_outputs = self.forward_generator_step(
            clean,
            noisy,
        )
        generator_outputs["one_labels"] = one_labels
        generator_outputs["clean"] = clean

        loss, _, _, _, _ = self.calculate_generator_loss(generator_outputs)

        discrim_loss_metric = self.calculate_discriminator_loss(generator_outputs)
        if discrim_loss_metric is None:
            discrim_loss_metric = torch.tensor([0.0])

        return loss.item(), discrim_loss_metric.item()

    @torch.no_grad()
    def save_enhance_samples(self, epoch, num_batches=1, num_samples=4):
        if not self.is_main:
            return
        out_dir = self.log_dir / f"enhance_epoch_{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for b_idx, batch in enumerate(self.test_ds):
            if b_idx >= num_batches:
                break
            clean = batch[0].to(self.gpu_id)
            noisy = batch[1].to(self.gpu_id)
            generator_outputs = self.forward_generator_step(clean, noisy)
            est_audio = self.peak_normalize(generator_outputs["est_audio"].detach().cpu())
            batch_size = min(num_samples, est_audio.size(0))
            for i in range(batch_size):
                length = est_audio.size(1)
                est = self.peak_normalize(est_audio[i, :length].unsqueeze(0))
                prefix = f"epoch{epoch}_b{b_idx}_idx{i}"
                torchaudio.save(
                    str(out_dir / f"{prefix}_enh.wav"), est, args.sample_rate
                )

    def test(self):
        self.model.eval()
        self.discriminator.eval()
        gen_loss_total = 0.0
        disc_loss_total = 0.0
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss
        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        template = "GPU: {}, Generator loss: {}, Discriminator loss: {}"
        logging.info(template.format(self.gpu_id, gen_loss_avg, disc_loss_avg))

        return gen_loss_avg

    def train(self):
        scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.decay_epoch, gamma=0.5
        )
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_disc, step_size=args.decay_epoch, gamma=0.5
        )
        best_train_loss = float("inf")
        best_gen_loss = float("inf")
        no_improve_train = 0
        no_improve_val = 0
        patience = 5
        start_epoch = 0
        if args.resume:
            ckpt = torch.load(args.resume, map_location=self.gpu_id)
            self.model.module.load_state_dict(ckpt["model"])
            self.discriminator.module.load_state_dict(ckpt["discriminator"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.optimizer_disc.load_state_dict(ckpt["optimizer_disc"])
            scheduler_G.load_state_dict(ckpt["scheduler_G"])
            scheduler_D.load_state_dict(ckpt["scheduler_D"])
            self.global_step = ckpt.get("global_step", 0)
            best_train_loss = ckpt.get("best_train_loss", best_train_loss)
            best_gen_loss = ckpt.get("best_gen_loss", best_gen_loss)
            no_improve_train = ckpt.get("no_improve_train", no_improve_train)
            no_improve_val = ckpt.get("no_improve_val", no_improve_val)
            start_epoch = ckpt.get("epoch", -1) + 1
            if self.gpu_id == 0:
                logging.info("Resumed from checkpoint %s at epoch %s", args.resume, start_epoch)
        for epoch in range(start_epoch, args.epochs):
            self.model.train()
            self.discriminator.train()
            train_loss_total = 0.0
            train_disc_loss_total = 0.0
            train_steps = 0
            for idx, batch in enumerate(self.train_ds):
                step = idx + 1
                loss, disc_loss, loss_mag, loss_ri, time_loss, gan_loss = self.train_step(
                    batch
                )
                train_loss_total += loss
                train_disc_loss_total += disc_loss
                train_steps += 1
                self.global_step += 1
                if self.is_main:
                    self.writer.add_scalar(
                        "train/loss_mag", float(loss_mag), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/loss_ri", float(loss_ri), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/time_loss", float(time_loss), self.global_step
                    )
                    self.writer.add_scalar(
                        "train/gan_loss", float(gan_loss), self.global_step
                    )
                template = "GPU: {}, Epoch {}, Step {}, loss: {}, disc_loss: {}"
                if (step % args.log_interval) == 0:
                    logging.info(
                        template.format(self.gpu_id, epoch, step, loss, disc_loss)
                    )
            train_loss_avg = train_loss_total / max(1, train_steps)
            train_disc_loss_avg = train_disc_loss_total / max(1, train_steps)
            gen_loss = self.test()
            if self.gpu_id == 0:
                logging.info("Epoch %s validation loss: %s", epoch, gen_loss)
            self._log_eval_step(
                epoch, epoch + 1, self.global_step, train_loss_avg, train_disc_loss_avg
            )
            if train_loss_avg < best_train_loss:
                best_train_loss = train_loss_avg
                no_improve_train = 0
            else:
                no_improve_train += 1

            if gen_loss < best_gen_loss:
                best_gen_loss = gen_loss
                no_improve_val = 0
            else:
                no_improve_val += 1
            path = os.path.join(
                args.save_model_dir,
                "CMGAN_epoch_" + str(epoch) + "_" + str(gen_loss)[:5],
            )
            if not os.path.exists(args.save_model_dir):
                os.makedirs(args.save_model_dir)
            if self.gpu_id == 0:
                torch.save(self.model.module.state_dict(), path)
                ckpt_path = os.path.join(args.save_model_dir, "ckpt_latest.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.module.state_dict(),
                        "discriminator": self.discriminator.module.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "optimizer_disc": self.optimizer_disc.state_dict(),
                        "scheduler_G": scheduler_G.state_dict(),
                        "scheduler_D": scheduler_D.state_dict(),
                        "global_step": self.global_step,
                        "best_train_loss": best_train_loss,
                        "best_gen_loss": best_gen_loss,
                        "no_improve_train": no_improve_train,
                        "no_improve_val": no_improve_val,
                    },
                    ckpt_path,
                )
            scheduler_G.step()
            scheduler_D.step()
            if self.is_main:
                self.save_enhance_samples(
                    epoch + 1, num_batches=1, num_samples=args.log_audio_samples
                )
            if dist.is_available() and dist.is_initialized() and self.gpu_id != 0:
                stop_flag = torch.tensor(0, device=self.gpu_id)
            else:
                stop_flag = torch.tensor(
                    1
                    if (no_improve_train >= patience or no_improve_val >= patience)
                    else 0,
                    device=self.gpu_id,
                )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                if self.gpu_id == 0:
                    logging.info(
                        "Early stopping at epoch %s after %s epochs without improvement.",
                        epoch,
                        patience,
                    )
                break


def main(rank: int, world_size: int, args):
    torch.cuda.set_device(rank)
    ddp_setup(rank, world_size)
    if rank == 0:
        print(args)
        available_gpus = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        print(available_gpus)
    train_ds, test_ds = dataloader.load_data(
        args.data_dir, args.batch_size, 2, args.cut_len
    )
    trainer = Trainer(train_ds, test_ds, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
