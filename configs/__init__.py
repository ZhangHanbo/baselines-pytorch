import os
import importlib

_cfg_list = os.listdir(".")

CONFIGS = {}
for cfg in _cfg_list:
    if not cfg.startswith("__init__"):
        alg = cfg.split("_")[0]
        env_id = cfg.split(".")[0].split("_")[1]
        if alg in CONFIGS:
            CONFIGS[alg][env_id] = importlib.import_module(cfg.split(".")[0])
        else:
            CONFIGS[alg] = {}
            CONFIGS[alg][env_id] = importlib.import_module(cfg.split(".")[0])

