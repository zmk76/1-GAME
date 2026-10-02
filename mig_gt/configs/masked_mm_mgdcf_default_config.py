# coding=utf-8

from dataclasses import dataclass
import typing
from mig_gt.utils.arg_utils import parse_bool, parse_int_list, parse_str_list
from mig_gt.configs.mm_mgdcf_default_config import MMMGDCFConfig


def load_masked_mm_mgdcf_default_config(dataset_name):
    if dataset_name == "baby":
        config = MMMGDCFConfig(
            lr=8e-3,          #1e-2->8e-3
            lr_decay=0.99,
            lr_decay_min=0.0,
            l2_coef=1e-3,       #1e-5->1e-3
            batch_size=8000,
            num_epochs=300,
            num_negs=1,
            patience=100,
            validation_freq=20,
            # validation_freq=5,
            embedding_size=256,     # 256 → 256
            feat_hidden_units=512,
            k_e=4,
            k_t=2,
            k_v=1,
            alpha=0.1,
            beta=0.9,
            input_feat_drop_rate=0.4,      #0.3->0.4
            feat_drop_rate=0.4,         #0.3->0.4
            user_x_drop_rate=0.3,
            item_x_drop_rate=0.3,
            edge_drop_rate=0.4,         #0.2->0.4
            z_drop_rate=0.2,
            mask_rate=0.0,
            use_dual=False,
            use_rp=True,
            bn=True,
            use_item_emb=False,
            num_clusters=10,         #5->10
            num_samples=20,         #10->20
        )
    elif dataset_name == "sports":
        config = MMMGDCFConfig(
            lr=5e-3,         #1e-2->5e-3
            lr_decay=0.99,
            lr_decay_min=1e-3,
            l2_coef=1e-5,      #1e-4->1e-5
            batch_size=8000,
            # num_epochs=800,#0,
            num_epochs=300,
            num_negs=1,
            # patience=100,
            patience=100,
            validation_freq=50,
            embedding_size=128,        #64->128
            feat_hidden_units=512,
            # feat_hidden_units=2048,

            # k=2,
            k_e=4,
            k_t=2,
            k_v=3,
            alpha=0.1,
            beta=0.9,
            # input_feat_drop_rate=0.2,
            # feat_drop_rate=0.2,

            input_feat_drop_rate=0.1,     #0.2->0.1
            feat_drop_rate=0.3,         #0.5->0.3

            user_x_drop_rate=0.1,
            item_x_drop_rate=0.2,
            edge_drop_rate=0.6,      #0.5->0.6
            z_drop_rate=0.2,
            mask_rate=0.0,
            use_dual=False,
            use_rp=True,
            bn=True,
            use_item_emb=False,
            num_clusters=3,    #5->3
            num_samples=20     #10->20
        )

    elif dataset_name == "clothing":
        config = MMMGDCFConfig(
            lr=1e-3,
            lr_decay=0.999,
            lr_decay_min=1e-4,
            l2_coef=1e-5,
            # l2_coef=1e-4,
            batch_size=8000,
            num_epochs=500,
            num_negs=1,
            patience=100,
            validation_freq=50,
            embedding_size=128,             #64->128
            feat_hidden_units=2048,
            # k=2,
            k_e=2,
            k_t=2,
            k_v=0,  # 0,
            alpha=0.1,
            beta=0.9,
            # input_feat_drop_rate=0.0,
            # feat_drop_rate=0.5,

            input_feat_drop_rate=0.1,
            feat_drop_rate=0.4,
            user_x_drop_rate=0.1,
            item_x_drop_rate=0.1,
            edge_drop_rate=0.25,
            z_drop_rate=0.3,
            mask_rate=0.0,
            use_dual=False,
            use_rp=True,
            bn=True,
            use_item_emb=False,
            num_clusters=0,
            num_samples=30      #20->30
        )

    return config




