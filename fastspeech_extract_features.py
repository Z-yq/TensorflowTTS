import logging
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataloaders import fastspeech_dataloader
from models.model import TTSmodel
from utils.user_config import UserConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

        self.dg = fastspeech_dataloader.FastSpeechDataLoader(self.config)

    def run(self, ):
        files_list = []
        out_data_dir = self.config['out_data_dir']
        os.makedirs(os.path.join(out_data_dir, 'data'), exist_ok=True)
        for batch in tqdm(self.dg.extractor()):
            charactors, durations, mels, tacotron_mels, speaker_ids, audios, names = batch
            decoder_output, mel_outputs, duration_preds = self.am_model.acoustic_model(
                input_ids=charactors,
                attention_mask=tf.math.not_equal(charactors, 0),
                speaker_ids=speaker_ids,
                duration_gts=durations,
                training=False
            )
            mel_outputs = mel_outputs.numpy()

            for i in range(len(charactors)):
                length = np.sum(np.where(durations[i] == 0, 0, 1))
                length = int(length)
                name = names[i]
                target_mel = mels[i]
                fastspeech_mel = mel_outputs[i][:len(target_mel)]

                audio = audios[i]

                tacotron_mel = tacotron_mels[i]
                dur = durations[i][:length]
                char = charactors[i][:length]
                np.savez(name, charactor=char,
                         tacotron_mel=tacotron_mel,
                         target_mel=target_mel,
                         duration=dur, audio=audio,
                         speaker=speaker_ids[i],
                         fastspeech_mel=fastspeech_mel,
                         )
                files_list.append(name)
        with open(os.path.join(out_data_dir, 'train.list'), 'w', encoding='utf-8') as f:
            for line in files_list:
                f.write(line + '\n')


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_config', type=str, default='./configs/common.yml', help='the am data config path')
    parse.add_argument('--model_config', type=str, default='./configs/fastspeech.yml', help='the am model config path')
    args = parse.parse_args()

    config = UserConfig(args.data_config, args.model_config)
    extract = Extractor(config)
    extract.run()
