from preprocess.seed import load_seed, seed_config
from preprocess.seed_v import load_seed_v, seed_v_config
from preprocess.seed_iv import load_seed_iv, seed_iv_config

load_dataset = {
    'SEED': lambda batch_size, shuffle, normalize: load_seed(batch_size, shuffle, normalize),
    'SEED-V': lambda batch_size, shuffle, normalize: load_seed_v(batch_size, shuffle, normalize),
    'SEED-IV': lambda batch_size, shuffle, normalize: load_seed_iv(batch_size, shuffle, normalize)
}

load_dataset_config = {
    'SEED': seed_config,
    'SEED-V': seed_v_config,
    'SEED-IV': seed_iv_config
}
