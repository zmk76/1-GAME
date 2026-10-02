from logging import getLogger
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.configurator import Config
import platform
import os
import pandas as pd
import numpy as np
import torch


def convert_freedom_dataset_to_common(split_dataset, num_users, mask_datasets):
    split_df = split_dataset.df

    user_field = split_dataset.config['USER_ID_FIELD']
    item_field = split_dataset.config['ITEM_ID_FIELD']

    # group by user_field

    user_item_edges = np.array(split_df[[user_field, item_field]].values, dtype=np.int64)

    # convert to dict user=>items
    user_items_dict = split_df.groupby(user_field)[item_field].apply(list).to_dict()
    for user_index in range(num_users):
        if user_index not in user_items_dict:
            user_items_dict[user_index] = []


    mask_dfs = [mask_dataset.df for mask_dataset in mask_datasets]
    mask_df = pd.concat(mask_dfs)

    mask_user_items_dict = mask_df.groupby(user_field)[item_field].apply(list).to_dict()
    for user_index in range(num_users):
        if user_index not in mask_user_items_dict:
            mask_user_items_dict[user_index] = []

    return user_item_edges, user_items_dict, mask_user_items_dict
    



def load_data(dataset):
    config_dict = {}
    config = Config("FREEDOM", dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    num_users = dataset.user_num
    num_items = dataset.item_num

    train_user_item_edges, train_user_items_dict, train_mask_user_items_dict = convert_freedom_dataset_to_common(train_dataset, num_users, [valid_dataset, test_dataset])
    valid_user_item_edges, valid_user_items_dict, valid_mask_user_items_dict = convert_freedom_dataset_to_common(valid_dataset, num_users, [train_dataset, test_dataset])
    test_user_item_edges, test_user_items_dict, test_mask_user_items_dict = convert_freedom_dataset_to_common(test_dataset, num_users, [train_dataset, valid_dataset])



    v_feat, t_feat = None, None
    if not config['end2end'] and config['is_multimodal_model']:
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        # if file exist?
        v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
        t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
        if os.path.isfile(v_feat_file_path):
            v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)
        if os.path.isfile(t_feat_file_path):
            t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)

        assert v_feat is not None or t_feat is not None, 'Features all NONE'


    # print("train")
    # print(train_user_items_dict[19439])
    # print(train_mask_user_items_dict[19439])
    # print("valid")
    # print(valid_user_items_dict[19439])
    # print(valid_mask_user_items_dict[19439])
    # print("test")
    # print(test_user_items_dict[19439])
    # print(test_mask_user_items_dict[19439])


    return train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat




import dgl

def dgl_add_all_reversed_edges(g):
    edge_dict = {}
    for etype in list(g.canonical_etypes):
        col, row = g.edges(etype=etype)
        edge_dict[etype] = (col, row)

        if etype[0] != etype[2]:
            new_etype = (etype[2], "r.{}".format(etype[1]), etype[0])
            edge_dict[new_etype] = (row, col)

    new_g = dgl.heterograph(edge_dict)

    for key in g.ndata:
        print("key = ", key)
        new_g.ndata[key] = g.ndata[key]

    return new_g






def build_hetero_graph(user_item_edges, num_users, num_items):
    edge_dict = {}

    user_item_edges = (user_item_edges[:, 0], user_item_edges[:, 1])
    edge_dict[("user", "user_item", "item")] = user_item_edges

    if True:

        item_image_edges = (torch.arange(num_items), torch.arange(num_items))
        edge_dict[("item", "item_image", "item_image")] = item_image_edges

        item_text_edges = (torch.arange(num_items), torch.arange(num_items))
        edge_dict[("item", "item_text", "item_text")] = item_text_edges

        g = dgl.heterograph(edge_dict, num_nodes_dict={"user": num_users, "item": num_items, "item_image": num_items, "item_text": num_items})
    else:
        g = dgl.heterograph(edge_dict, num_nodes_dict={"user": num_users, "item": num_items})

    g = dgl_add_all_reversed_edges(g)

    # compute deg for each node


    return g





def load_hetero_data(dataset):
    train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat = load_data(dataset)

    train_hetero_g = build_hetero_graph(train_user_item_edges, num_users, num_items)
    valid_hetero_g = build_hetero_graph(valid_user_item_edges, num_users, num_items)
    test_hetero_g = build_hetero_graph(test_user_item_edges, num_users, num_items)

    return train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat, train_hetero_g, valid_hetero_g, test_hetero_g


