import os
import torch
from typing import List, Optional, Union, Dict
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy
from transformers.tokenization_utils_base import EncodedInput, BatchEncoding



class SPTokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.eos_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def tokenize(self, s: str):
        return self.sp_model.EncodeAsPieces(s)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def decode_tokens(self, tokens: List[str]) -> str:
        text = self.sp_model.DecodePieces(tokens)
        return text

    def convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.index_special_tokens:
            return ""
        return self.sp_model.IdToPiece(index)

sptokenizer = SPTokenizer(model_path="raw_model/chinese_vocab/tokenizer.model")



from tqdm import tqdm

all_list = []

for i in tqdm(range(70000)):
    try:
        v = sptokenizer.convert_id_to_token(i)
        all_list.extend([v])
    except Exception as e:
        break

len(all_list)


def _is_chinese_char(cp):
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return False
    return True

chinese_list = list(set(filter(is_chinese, all_list)))

import pandas as pd 

data = pd.DataFrame({'word':chinese_list}).pipe(
    lambda x: x.assign(**{
        'len':x['word'].apply(lambda j: len(j))
    }).query('len > 0')
)
data.pipe(
    lambda x:x.groupby(['len']).agg(
        freq = ('word', 'count')
    )
)

data.query('len <= 10').sort_values(by=['len'], ascending=False)
chinese_list_finally = data.query('len <= 10')['word'].tolist()



from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

old_model_name_or_path = "raw_model/mpt-7b-chat"

old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path)
old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path, trust_remote_code=True)

old_tokenizer.add_tokens(chinese_list_finally)
# len(old_tokenizer)  80889
# int(len(old_tokenizer) // 64 + 1) * 64  80896

#old_model.get_input_embeddings() Embedding(50432, 4096)

old_model.resize_token_embeddings(80896)  # Embedding(80896, 4096)

new_model_name_or_path = "mpt_chat_7b_chinese_no_train"

old_tokenizer.save_pretrained(new_model_name_or_path)
old_model.save_pretrained(new_model_name_or_path,max_shard_size="4GB")