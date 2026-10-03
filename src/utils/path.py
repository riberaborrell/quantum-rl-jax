import os

# absolute path of the src directory
SRC_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# absolute path of the data directory
DATA_DIR_PATH = os.path.join(SRC_DIR_PATH, 'data')


def make_dir_path(dir_path):
    ''' Create directories of the given path if they do not already exist
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def get_env_dir_path(env_name):
    ''' Get (and create) the data directory of the given environment, i.e. data/[env_name]
    '''
    dir_path = os.path.join(DATA_DIR_PATH, env_name)
    make_dir_path(dir_path)
    return dir_path

def get_algorithm_dir_path(env_name, algorithm_name):
    ''' Get (and create) the data directory of the given algorithm, i.e. data/[env_name]/[algorithm_name]
    '''
    dir_path = os.path.join(get_env_dir_path(env_name), algorithm_name)
    make_dir_path(dir_path)
    return dir_path

def get_q_value_dir_path(env_name, algorithm_name):
    ''' Get (and create) the q-value directory of the given algorithm, i.e. data/[env_name]/[algorithm_name]/q-value
    '''
    dir_path = os.path.join(get_algorithm_dir_path(env_name, algorithm_name), 'q-value')
    make_dir_path(dir_path)
    return dir_path
