import logging
import os

import librosa
import numpy as np
import tensorflow as tf
from utils.speech_featurizers import SpeechFeaturizer



class VocoderDataLoader():
    def __init__(self, config, training=True):
        self.config = config
        self.batch = config['batch_size']
        self.make_file_list(self.config['train_list'] if training else self.config['eval_list'], training=training)
        self.epochs = 1
        self.steps = 0
        self.speech_featurizer=SpeechFeaturizer(config)

        self._target_pad = -(self.config['max_abs_value'] + 0.1)
    def make_file_list(self, wav_list, training=True):
        with open(wav_list, encoding='utf-8') as f:
            data = f.readlines()
        data = [i.strip() for i in data if i != '']
        num = len(data)
        if training:
            self.train_list = data[:int(num * 0.95)]
            self.test_list = data[int(num * 0.95):]
            np.random.shuffle(self.train_list)
            self.train_offset = 0
            self.test_offset = 0
            logging.info('load train list {} test list{}'.format(len(self.train_list), len(self.test_list)))
        else:
            self.test_list = data
            self.offset = 0

    def get_per_epoch_steps(self):
        return len(self.train_list) // self.batch

    def eval_per_epoch_steps(self):
        return len(self.test_list) // self.batch

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

    def return_data_types(self):

        return (tf.float32, tf.float32)


    def return_data_shape(self):

        return (
            tf.TensorShape([None, None, self.config['num_mels']]) ,
            tf.TensorShape([None,None,1]),
        )

    def generate(self,train=True):
        y = []
        x = []
        maxnum = self.config['frame_length']

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
            if self.config['load_from_npz']:
                try:
                    data=np.load(line.strip())

                    target_wav=data['audio']
                    if self.config['adjust_type']=='tacotron':
                        train_mel=data['tacotron_mel']
                    elif self.config['adjust_type']=='fastspeech':
                        train_mel = data['fastspeech_mel']
                    else:
                        train_mel = data['target_mel']

                except:
                    logging.info('{} load data failed , skip'.format(line))
                    continue
                if len(target_wav) > maxnum:
                    pick = np.random.randint(0, len(target_wav) - maxnum, 1)[0]
                    target_wav = target_wav[pick:pick + maxnum]
                    pick_mel=pick//self.speech_featurizer.hop_size
                    max_mel=maxnum//self.speech_featurizer.hop_size
                    train_mel=train_mel[pick_mel:pick_mel+max_mel]


                y.append(target_wav)
                x.append(train_mel)
                if len(y) == self.batch:
                    break
            else:
                try:

                    target_wav=self.speech_featurizer.load_wav(line.strip())
                except:
                    logging.info('{} load data failed , skip'.format(line))
                    continue
                if len(target_wav) > maxnum:
                    pick = np.random.randint(0, len(target_wav) - maxnum, 1)[0]
                    target_wav = target_wav[pick:pick + maxnum]
                y.append(target_wav)
                if len(y)==self.batch:
                    break
        if maxnum % self.config['num_mels'] != 0:
            maxnum += self.config['num_mels'] - maxnum % self.config['num_mels']
        y=self.speech_featurizer.pad_signal(y,maxnum)
        if self.config['load_from_npz']:
            x=self._prepare_targets(x,maxnum//self.speech_featurizer.hop_size)
        else:
            process_wav=np.hstack(y)
            mel= self.speech_featurizer.melspectrogram(process_wav)

            x=mel.reshape([len(y),-1,self.config['num_mels']])

        x = np.array(x,'float32')
        y = np.array(y,'float32')
        return x, y[:, :, np.newaxis]
    def generator(self, train=True):
        while 1:
            x,y= self.generate(train)
            if x.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue
            yield x,y

    def _prepare_targets(self, targets,max_len=None):
        if max_len is None:
            max_len = max([len(t) for t in targets])
        new=[]
        for t in targets:
            if t.shape[0]>=max_len:
                new.append(t[:max_len])
            else:
                new.append(self._pad_target(t,max_len))

        return np.stack(new)
    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)