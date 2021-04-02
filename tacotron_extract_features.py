import logging
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataloaders import tacotron_dataloader
from models.model import TTSmodel
from utils.user_config import UserConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
gpus = tf.config.experimental.list_physical_devices('GPU')
logging.info('valid gpus:%d' % len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Extractor():
    def __init__(self, config):
        self.config = config
        self.am_model = TTSmodel(config=config)
        self.am_model.load_model(True)
        self.am_model.acoustic_model._build()
        self.checkpoint_dir = os.path.join(self.config["outdir"], "checkpoints")
        files = os.listdir(self.checkpoint_dir)
        files.sort(key=lambda x: int(x.split('_')[-1].replace('.h5', '')))
        self.am_model.acoustic_model.load_weights(os.path.join(self.checkpoint_dir, files[-1]))
        logging.info('acoustic load model at {}'.format(os.path.join(self.checkpoint_dir, files[-1])))

        self.dg = tacotron_dataloader.TacotronDataLoader(self.config)

    def run(self, ):
        files_list=[]
        out_data_dir = self.config['out_data_dir']
        os.makedirs(os.path.join(out_data_dir, 'data'), exist_ok=True)
        for batch in tqdm(self.dg.extractor()):
            charactor, char_length, mel, mel_length, speaker, audios, names = batch
            decoder_output, mel_outputs, stop_token_prediction, alignment_history = self.am_model.acoustic_model(
                input_ids=charactor,
                input_lengths=char_length,
                speaker_ids=speaker,
                mel_outputs=mel,
                mel_lengths=mel_length,
                maximum_iterations=2000,
                use_window_mask=False,
                win_front=2,
                win_back=3,
                training=False
            )
            mel_outputs = mel_outputs.numpy()
            durations = alignment_history.numpy()
            stop_token_prediction = tf.nn.sigmoid(stop_token_prediction)

            for idx in range(len(char_length)):
                stop = stop_token_prediction[idx]
                stop = stop.numpy()
                stop = np.where(stop > 0.6, 1, 0)
                stop = stop.flatten().tolist()
                if 1 in stop:
                    num = stop.index(1) + 1
                else:
                    num = len(stop)
                num //= self.config['outputs_per_step']
                durations[idx, :, num:] = 0.

            durations = np.sum(durations, -1) * self.config['outputs_per_step']
            durations = np.rint(durations)

            for i in range(len(char_length)):
                name = names[i]
                char_l = char_length[i]
                mel_l = mel_length[i]

                audio = audios[i]
                T = int(mel_l * self.config['hop_size'] * self.config['sample_rate'])
                audio = audio[:T]
                target_mel = mel[i][:mel_l]
                tacotron_mel = mel_outputs[i][:mel_l]
                dur = durations[i][:char_l]
                char = charactor[i][:char_l]
                spk=speaker[i]

                np.savez(os.path.join(out_data_dir,'data', str(spk)+'_'+name + '.npz'), charactor=char,
                         tacotron_mel=tacotron_mel,
                         target_mel=target_mel,
                         duration=dur, audio=audio,
                         speaker=spk
                         )
                files_list.append(os.path.join(out_data_dir,'data',  str(spk)+'_'+name + '.npz'))
        with open(os.path.join(out_data_dir,'train.list'),'w',encoding='utf-8') as f:
            for line in files_list:
                f.write(line+'\n')

if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/common.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/tacotron.yml', help='the am model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config, args.model_config)
    extract = Extractor(config)
    extract.run()
