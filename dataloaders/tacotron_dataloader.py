import os
import numpy as np
from utils.speech_featurizers import SpeechFeaturizer
from utils.text_featurizers import TextFeaturizer
import tensorflow as tf
import logging

class TacotronDataLoader():
    def __init__(self,config,training=True):
        self.speech_featurizer=SpeechFeaturizer(config)
        self.text_featurizer=TextFeaturizer(config)
        self.config=config
        self.batch = config['batch_size']
        self.make_file_list(self.config['train_list'] if training else self.config['eval_list'], training=training)
        self.min_value = -self.config['max_abs_value']
        self._target_pad = -(self.config['max_abs_value']+0.1)
        self._token_pad = 1.
        self.epochs=1
        self.steps=0

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
            if self.config['balance_spk_utts']:
                spk_utt={}
                for line in self.train_list:
                    a,b,c=line.strip().split('\t')
                    if c in spk_utt:
                        spk_utt[c].append(line)
                    else:
                        spk_utt[c]=[line]
                maxlen=max([len(spk_utt[i]) for i in spk_utt])
                self.train_list=[]
                for key in spk_utt:
                    datas=spk_utt[key]
                    if len(datas)<maxlen:
                        factor=int(np.rint(maxlen/len(datas)))
                    else:
                        factor=1
                    datas*=factor
                    self.train_list+=datas
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
        #charactor, char_length, mel, mel_length, stop_gts, speaker, guided_attention
        return (tf.int32,tf.int32,tf.float32,tf.int32, tf.float32,tf.int32,tf.float32)


    def return_data_shape(self):
        # charactor, char_length, mel, mel_length, stop_gts, speaker, guided_attention
        return (
            tf.TensorShape([None, None]) ,
            tf.TensorShape([None,]),
            tf.TensorShape([None,None,self.config['num_mels']]),
            tf.TensorShape([None,]),
            tf.TensorShape([None,None]),
            tf.TensorShape([None,None]),
            tf.TensorShape([None,None,None]),
        )
    def GuidedAttention(self, N, T, g=0.5):
        W = np.zeros((N, T), dtype=np.float32)
        for n in range(N):
            for t in range(T):
                W[n, t] = 1 - np.exp(-(t / float(T) - n / float(N)) ** 2 / (2 * g * g))
        return W

    def make_Att_targets(self, input_length, targets_length, inputs_shape, mel_target_shape):
        att_targets = []
        att_mask = []
        mel_target_shape //= self.config['outputs_per_step']
        for i, j in zip(input_length, targets_length):
            # i=inputs_shape
            step = int(j / self.config['outputs_per_step'])
            pad = np.zeros([inputs_shape, mel_target_shape])
            pad[i:, :step] = 1
            maskpad = np.zeros([inputs_shape, mel_target_shape])
            maskpad[:, :step] = 1
            att_target = self.GuidedAttention(i, step, 0.1)

            pad[:att_target.shape[0], :att_target.shape[1]] = att_target
            att_targets.append(pad)
            att_mask.append(maskpad)
        att_targets = np.array(att_targets)
        att_mask = np.array(att_mask)
        return att_targets.astype('float32'), att_mask.astype('float32')

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
            charactor, char_length, mel, mel_length, speaker = [], [], [], [], []
            audios=[]
            names=[]
            for i in range(self.batch * 10):

                line = data[self.train_offset]
                self.train_offset += 1
                if self.train_offset > len(data) - 1:
                    break

                wav_path, text, spkid = line.strip().split('\t')
                try:
                    wav = self.speech_featurizer.load_wav(wav_path)
                    target_mel = self.speech_featurizer.melspectrogram(wav)
                # print(target_mel.shape)
                except:
                    logging.info('{} load data failed , skip'.format(wav_path))
                    continue
                try:
                    text_tokens = self.text_featurizer.extract(text)
                except:
                    logging.info('{} to token failed,skip'.format(text))
                    continue
                try:
                    speaker_id = self.text_featurizer.spker_map[spkid]
                except:
                    logging.info('{} not in spk map,skip'.format(spkid))
                    continue
                audios.append(wav)
                names.append(os.path.split(wav_path)[-1].replace('.wav',''))

                charactor.append(np.array(text_tokens))
                char_length.append(len(text_tokens))
                mel.append(target_mel)
                mel_length.append(len(target_mel))

                speaker.append([speaker_id])
                if len(charactor) == self.batch:
                    break
            output_per_step = self.config['outputs_per_step']
            charactor = self._prepare_inputs(charactor)
            char_length = np.array(char_length, 'int32')
            mel = self._prepare_targets(mel, output_per_step)
            mel_length = np.array(mel_length, 'int32')

            speaker = np.array(speaker, 'int32')
            T = mel.shape[1] * self.speech_featurizer.hop_size
            audios=tf.keras.preprocessing.sequence.pad_sequences(audios,T,'float32','post','post')
            yield charactor, char_length, mel, mel_length,  speaker,audios,names
    def generate(self, train=True):
        charactor, char_length, mel, mel_length, stop_gts, speaker=[],[],[],[],[],[]


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
            wav_path,text,spkid=line.strip().split('\t')
            try:
                wav = self.speech_featurizer.load_wav(wav_path)
                target_mel = self.speech_featurizer.melspectrogram(wav)
            # print(target_mel.shape)
            except:
                logging.info('{} load data failed , skip'.format(wav_path))
                continue
            try:
                text_tokens=self.text_featurizer.extract(text)
            except:
                logging.info('{} to token failed,skip'.format(text))
                continue
            try:
                speaker_id=self.text_featurizer.spker_map[spkid]
            except:
                logging.info('{} not in spk map,skip'.format(spkid))
                continue
            token_target = np.asarray([0.] * (len(target_mel) - 1))
            charactor.append(np.array(text_tokens))
            char_length.append(len(text_tokens))
            mel.append(target_mel)
            mel_length.append(len(target_mel))
            stop_gts.append(token_target)
            speaker.append([speaker_id])
            if len(charactor) == self.batch:
                break
        output_per_step=self.config['outputs_per_step']
        charactor=self._prepare_inputs(charactor)
        char_length=np.array(char_length,'int32')
        mel=self._prepare_targets(mel,output_per_step)

        mel_length=np.array(mel_length,'int32')
        stop_gts=self._prepare_token_targets(stop_gts,output_per_step)
        speaker=np.array(speaker,'int32')

        return charactor, char_length, mel, mel_length, stop_gts, speaker


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
            charactor, char_length, mel, mel_length, stop_gts, speaker= self.generate(train)

            if charactor.shape[0] == 0:
                logging.info('load data length zero,continue')
                continue
            guide_matrix,_=self.make_Att_targets(char_length,mel_length,np.max(char_length),np.max(mel_length))
            yield charactor.astype('int32'), char_length.astype('int32'), mel.astype('float32'), mel_length.astype('int32'), stop_gts.astype('float32'), speaker.astype('int32')\
                ,guide_matrix.astype('float32')