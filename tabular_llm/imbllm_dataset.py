import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class ImbLLMDataset(Dataset):
    def get_ds_size(self, ds_len, no_cont):
        # dataset length
        self.ds_len = ds_len
        # number of continuous variables
        self.no_cont = no_cont
        
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        # If int, what else?
        row = self._data.fast_slice(key, 1)

        # original dataset to train LLM
        if key < self.ds_len:
            # permute only feature variables
            all_column_idx = list(range(row.num_columns))
            shuffle_idx = all_column_idx[:-1]
            random.shuffle(shuffle_idx)
            # keep target variable at the beginning
            target_idx = all_column_idx[-1]

            shuffled_text = ", ".join(
                ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in shuffle_idx]
            )
            shuffled_text = "{} is {}, ".format(row.column_names[target_idx],
                                                str(row.columns[target_idx].to_pylist()[0]).strip()) + shuffled_text
        else: # SMOTE dataset to train LLM with only interpolated continuous variables
            # permute only continuous feature variables
            cont_column_idx = list(range(self.no_cont))
            random.shuffle(cont_column_idx)
            # keep target variable at the beginning
            target_idx = row.num_columns - 1

            shuffled_text = ", ".join(
                ["%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip()) for i in cont_column_idx]
            )
            shuffled_text = "{} is {}, ".format(row.column_names[target_idx],
                                                str(row.columns[target_idx].to_pylist()[0]).strip()) + shuffled_text

        # print("shuffled_text: {}".format(shuffled_text))
        tokenized_text = self.tokenizer(shuffled_text)

        return tokenized_text

    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """ GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()

        return batch

