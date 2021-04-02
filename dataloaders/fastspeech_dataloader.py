import logging
import os

import numpy as np
import tensorflow as tf

from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer


class FastSpeechDataLoader():
    def __init__(self, config, training=True):
        self.speech_featurizer = SpeechFeaturizer(config)
        self.text_featurizer = TextFeaturizer(config)
        self.config = config
        self.batch = config['batch_size']
        self.make_file_list(self.config['train_list'] if training else self.config['eval_list'], training=training)
        self.min_value = -self.config['max_abs_value']
        self._target_pad = -(self.config['max_abs_value'] + 0.1)
        self._token_pad = 1.
        self.epochs=1

    def make_file_list(self, wav_list, training=True):
        with open(wav_list, encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        num = len(data)
        if training:
            self.train_list = data[:int(num * 0.98)]
            self.test_list = data[int(num * 0.98):]
            np.random.shuffle(self.train_list)
            self.train_offset = 0
            self.test_offset = 0
            logging.info('load train list {} test list{}'.format(len(self.train_list), len(self.test_list)))
            if self.config['balance_spk_utts']:
                spk_utt = {}
                for line in self.train_list:
                    c = os.path.split(line)[-1][:4]
                    if c in spk_utt:
                        spk_utt[c].append(line)
                    else:
                        spk_utt[c] = [line]
                maxlen = max([len(spk_utt[i]) for i in spk_utt])
                self.train_list = []
                for key in spk_utt:
                    datas = spk_utt[key]
                    if len(datas) < maxlen:
                        factor = int(np.rint(maxlen / len(datas)))
                    else:
                        factor = 1
                    datas *= factor
                    self.train_list += datas
                np.random.shuffle(self.train_list)
                logging.info('balance spk utts: train list {}'.format(len(self.train_list)))
        else:
            self.test_list = data
            self.offset = 0

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_list) // self.batch

    def return_data_types(self):
        # charactor, duration, mel,tacotron_mel, speaker_id
        return (tf.int32, tf.int32, tf.float32, tf.float32, tf.int32)

    def return_data_shape(self):
        # charactor, duration, mel,tacotron_mel, speaker_id
        return (
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None, self.config['num_mels']]),
            tf.TensorShape([None, None, self.config['num_mels']]),
            tf.TensorShape([None, None]),

        )

    def load_state(self, outdir):
        try:

            dg_state = np.load(os.path.join(outdir, 'dg_state.npz'))

            self.epochs = int(dg_state['epoch'])
            self.train_offset = int(dg_state['train_offset'])
            train_list = dg_state['train_list'].tolist()
            if len(train_list) != len(self.train_list):
                logging.info('history train list not equal new load train list ,data loader use init state')
                self.epochs = 0
                self.train_offset = 0
        except FileNotFoundError:
            logging.info('not found state file,init state')
        except:
            logging.info('load state falied,use init state')

    def save_state(self, outdir):

        np.savez(os.path.join(outdir, 'dg_state.npz'), epoch=self.epochs, train_offset=self.train_offset,
                 train_list=self.train_list)
    def extractor(self,):
        data=self.train_list+self.test_list
        while self.train_offset<len(data):
            charactors, durations, mels, tacotron_mels, speaker_ids ,audios,names= [], [], [], [], [],[],[]

            for i in range(self.batch * 10):

                line = self.train_list[self.train_offset]
                self.train_offset += 1
                if self.train_offset > len(data) - 1:
                   break

                try:
                    data = np.load(line.strip())

                except:
                    logging.info('{} load data failed , skip'.format(line))
                    continue
                charactor = data['charactor']
                tacotron_mel = data['tacotron_mel']
                target_mel = data['target_mel']
                duration = data['duration']
                spk = data['speaker']
                audio=data['audio']

                if np.sum(duration) > len(target_mel):
                    diff = np.sum(duration) - len(target_mel)
                    diff = int(diff)
                    for i in range(1, diff + 1):
                        index = np.where(duration == duration.max())[0][0]
                        duration[index] -= 1
                elif np.sum(duration) < len(target_mel):
                    diff = len(target_mel) - np.sum(duration)
                    diff = int(diff)
                    for i in range(1, diff + 1):
                        index = np.where(duration[:-1] == duration[:-1].min())[0][0]
                        duration[index] += 1
                charactors.append(charactor)
                tacotron_mels.append(tacotron_mel)
                mels.append(target_mel)
                durations.append(duration)
                speaker_ids.append(spk)
                audios.append(audio)
                names.append(line.strip())
                if len(charactors) == self.batch:
                    break

            charactors = self._prepare_inputs(charactors)
            # mels = self._prepare_targets(mels, 1)
            # tacotron_mels = self._prepare_targets(tacotron_mels, 1)
            # mel_length = np.array(mel_length, 'int32')
            durations = self._prepare_inputs(durations)
            speaker_ids = np.array(speaker_ids, 'int32')
            durations = durations.astype('int32')
            yield charactors, durations, mels, tacotron_mels, speaker_ids,audios,names
    def generate(self, train=True):
        charactors, durations, mels,tacotron_mels, speaker_ids = [], [], [], [],[]

        for i in range(self.batch * 10):
            if train:
                line = self.train_list[self.train_offset]
                self.train_offset += 1
                if self.train_offset > len(self.train_list) - 1:
                    self.train_offset = 0
                    np.random.shuffle(self.train_list)
                    self.epochs += 1
            else:
                line = self.test_list[self.test_offset]
                self.test_offset += 1
                if self.test_offset > len(self.test_list) - 1:
                    self.test_offset = 0

            try:
                data = np.load(line.strip())

            except:
                logging.info('{} load data failed , skip'.format(line))
                continue
            charactor = data['charactor']
            tacotron_mel = data['tacotron_mel']
            target_mel =data['target_mel']
            duration =data['duration']
            spk =data['speaker']
            if np.sum(duration)>len(target_mel):
                diff=np.sum(duration)-len(target_mel)
                diff= int(diff)
                for i in range(1,diff+1):
                    index=np.where(duration==duration.max())[0][0]
                    duration[index]-=1
            elif np.sum(duration)<len(target_mel):
                diff=len(target_mel)-np.sum(duration)
                diff = int(diff)
                for i in range(1,diff+1):
                    index = np.where(duration[:-1] == duration[:-1].min())[0][0]
                    duration[index] += 1
            charactors.append(charactor)
            tacotron_mels.append(tacotron_mel)
            mels.append(target_mel)
            durations.append(duration)
            speaker_ids.append(spk)
            if len(charactors) == self.batch:
                break

        charactors = self._prepare_inputs(charactors)

        mels = self._prepare_targets(mels, 1)
        tacotron_mels = self._prepare_targets(tacotron_mels, 1)
        # mel_length = np.array(mel_length, 'int32')
        durations = self._prepare_inputs(durations)
        speaker_ids = np.array(speaker_ids, 'int32')
        durations=durations.astype('int32')
        return charactors, durations, mels, tacotron_mels, speaker_ids

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs])

    def _prepare_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets])
        return np.stack([self._pad_target(t, self._round_up(max_len, alignment)) for t in targets])

    def _prepare_token_targets(self, targets, alignment):
        max_len = max([len(t) for t in targets]) + 1
        return np.stack([self._pad_token_target(t, self._round_up(max_len, alignment)) for t in targets])

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self.text_featurizer.pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

    def _pad_token_target(self, t, length):
        return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def generator(self, train=True):
        while 1:
            charactors, durations, mels, tacotron_mels, speaker_ids = self.generate(train)
            if charactors.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue

            yield  charactors, durations, mels, tacotron_mels, speaker_ids
