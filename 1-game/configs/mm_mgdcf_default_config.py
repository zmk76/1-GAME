# coding=utf-8

from dataclasses import dataclass
import typing
from mig_gt.utils.arg_utils import parse_bool, parse_int_list, parse_str_list


@dataclass
class MMMGDCFConfig(object):
    lr: float
    lr_decay: float
    lr_decay_min: float
    l2_coef: float
    batch_size: int
    num_epochs: int

    num_negs: int

    patience: int
    validation_freq: int

    embedding_size: int
    feat_hidden_units: int
    # k: int

    k_e: int
    k_t: int
    k_v: int

    alpha: float
    beta: float

    input_feat_drop_rate: float
    feat_drop_rate: float
    user_x_drop_rate: float
    item_x_drop_rate: float
    edge_drop_rate: float
    z_drop_rate: float

    use_dual: bool

    use_rp: bool = False
    mask_rate: float = 0.0
    bn: bool = True
    use_item_emb: bool = False

    num_clusters: int = 0

    num_samples: int = 0




