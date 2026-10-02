# coding=utf-8

from dataclasses import dataclass
import typing
from mig_gt.utils.arg_utils import parse_bool, parse_int_list, parse_str_list


def combine_args_into_config(config, args):
    for key, value in args.__dict__.items():
        if value is not None and hasattr(config, key):
            old_value = getattr(config, key)
            setattr(config, key, value)
            print("set config.{} based on args.{}: {} => {}".format(key, key, old_value, value))

    return config


def add_arguments_by_config_class(parser, config_class):
    
    field_type_dict = typing.get_type_hints(config_class)

    for field_name, field_type in field_type_dict.items():
        arg_type = field_type

        if field_type == typing.List[str]:
            arg_type = parse_str_list
        elif field_type == typing.List[int]:
            arg_type = parse_int_list
        elif field_type == bool:
            arg_type = parse_bool
        
        parser.add_argument("--{}".format(field_name), type=arg_type, required=False)

        print("generate argument --{} with type {}".format(field_name, arg_type))
    
    return parser



