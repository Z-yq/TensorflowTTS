
import codecs
import tensorflow as tf
from utils.tools import preprocess_paths
from utils.normalize import NSWNormalizer
import pypinyin as ppy
import logging


class TextFeaturizer:
    """
    Extract text feature based on char-level granularity.
    By looking up the vocabulary table, each line of transcript will be
    converted to a sequence of integer indexes.
    """

    def __init__(self, config: dict,show=False):
        """
        config = {
            "vocabulary": str,
            "blank_at_zero": bool,
            "beam_width": int,
            "lm_config": {
                ...
            }
        }
        """
        self.config = config
        self.normlizer=NSWNormalizer
        self.config["vocabulary"] = preprocess_paths(self.config["vocabulary"])
        self.config["spker"] = preprocess_paths(self.config["spker"])
        self.config["maplist"] = preprocess_paths(self.config["maplist"])
        with open(self.config['spker']) as f:
            spks=f.readlines()
        self.spker_map={}
        for idx,spk in enumerate(spks):
            self.spker_map[spk.strip()]=idx
        with open(self.config["maplist"], encoding='utf-8') as f:
            data = f.readlines()
        self.map_dict = {}
        for line in data:
            try:
                a, b = line.strip().split('\t')
            except:
                content = line.split(' ')
                a = content[0]
                b = ' '.join(content[1:])
            a = a.replace('[', '').replace(']', '')
            b = b.split(' ')
            self.map_dict[a] = b

        self.num_classes = 0
        lines = []
        with codecs.open(self.config["vocabulary"], "r", "utf-8") as fin:
            lines.extend(fin.readlines())
        if show:
            logging.info('load token at {}'.format(self.config['vocabulary']))
        self.token_to_index = {}
        self.index_to_token = {}
        self.vocab_array = []
        self.tf_vocab_array = tf.constant([], dtype=tf.string)
        self.index_to_unicode_points = tf.constant([], dtype=tf.int32)
        index = 0
        if self.config["blank_at_zero"]:
            self.blank = 0
            index = 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [""]], axis=0)
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, [0]], axis=0)
        for line in lines:
            line = line.strip()  # Strip the '\n' char
            if line.startswith("#") or not line or line == "\n":
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.vocab_array.append(line)
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [line]], axis=0)
            upoint = tf.strings.unicode_decode(line, "UTF-8")
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, upoint], axis=0)
            index += 1
        self.num_classes = index
        if not self.config["blank_at_zero"]:
            self.blank = index
            self.num_classes += 1
            self.tf_vocab_array = tf.concat([self.tf_vocab_array, [""]], axis=0)
            self.index_to_unicode_points = tf.concat(
                [self.index_to_unicode_points, [0]], axis=0)
        self.stop=self.endid()
        self.pad=self.blank
        self.stop=-1

    def endid(self):
        return self.token_to_index['/S']


    def extract(self, text):
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints
        """
        text=self.normlizer(text).normalize()
        pinyins=ppy.pinyin(text,8,neutral_tone_with_five=True)
        tokens = []
        for py in pinyins:
            if py[0] in self.map_dict:
                tokens += self.map_dict[py[0]]
            else:
                if len(py[0]) > 1:
                    py = list(py[0])
                tokens += py

        feats=[self.token_to_index[token] for token in tokens]+[self.endid()]
        return feats

    def iextract(self,indx):
        texts=[self.index_to_token[idx] for idx in indx ]
        return texts

    def pad_texts(self,inps,max_length):
        inps = tf.keras.preprocessing.sequence.pad_sequences(inps, max_length, 'float32', 'post', 'post')
        return inps