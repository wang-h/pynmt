import os
import re
import sys
import logging
import argparse
from configparser import ConfigParser

from Utils.args import make_parser
from Utils.log import trace
from Utils.log import set_logging



def check_save_path(path):
    save_path = os.path.abspath(path)
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class Config(object):

    def __init__(self, prefix, training=True):
        self.training = training
        self.parser = make_parser(training) 
        self.args = self.parser.parse_args() 
        
        self.config_file = self.args.config
        self.read_config(self.config_file)
        self.filter()
        check_save_path(self.args.save_log)
        check_save_path(self.args.save_vocab)
        check_save_path(self.args.save_model)
        set_logging(prefix, self.args.save_log)
        
    def filter(self):
        if not self.args.use_cpu:
            del self.args.use_cpu
        

    def __getattr__(self, name):
        return getattr(self.args, name)

    def set_defaults(self, config, section):
        defaults = dict(config.items(section))
        for key, val in defaults.items():
            if key == "use_gpu":
                val = eval(val)
            else:
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(config[section], attr)(key)
                        break
                    except: pass
            defaults[key] = val
        
        self.parser.set_defaults(**defaults)
        self.args = self.parser.parse_args() 

    def check_config_exist(self):
        if not os.path.isfile(self.config_file):
            trace("""# Cannot find the configuration file. 
                {} does not exist! Please check.""".format(self.config_file))
            sys.exit(1)

    def read_config(self, config_file):
        self.check_config_exist()
        config = ConfigParser()
        config.read(config_file)

        groups = set(group.title for group in self.parser._action_groups)
        sections = groups.intersection(set(config.sections()))
        for section in sections:
            #print("#section", section)
            self.set_defaults(config, section)    
        
        save_path = os.path.abspath(self.args.save_model)
        dirname = os.path.dirname(save_path)
        
        config_file_bak = os.path.join(dirname, 'config.ini')
        check_save_path(config_file_bak)
        if not os.path.isfile(config_file_bak):
            with open(config_file_bak, 'w') as configfile:
                config.write(configfile)
    def __repr__(self):
        ret = "\n"
        pattern = r'\<class \'(.+)\'\>'
        for key, value in vars(self.args).items():
            class_type = re.search(pattern, str(type(value))).group(1)
            class_type = "[{}]".format(class_type)
            value_string = str(value)
            if len(value_string) > 80:
                value_string = "/".join(value_string.split("/")[:2]) +\
                    "/.../" + "/".join(value_string.split("/")[-2:])
            ret += "  {}\t{}\t{}\n".format(key.ljust(15),
                                           class_type.ljust(8), value_string)
        return ret

    # def config_device(config):
    #     if config.device_ids:
    #         config.device_ids = [int(x) for x in config.device_ids.split(",")]
    #     else:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    #             [str(idx) for idx in list(
    #                 range(config.gpu_ids, config.gpu_ids + config.num_gpus))])
    #         config.device_ids = list(range(config.num_gpus))
