import logging.config
import os
import shutil
import yaml
import sys
import time
from copy import deepcopy

def get_magic_fovs():
    return ['C%03dR%03d' % (col, row) for col in [2, 7] for row in [8, 23, 38]]

def full_traceback(func):
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            msg += 'args:\n'
            for arg in args:
                msg += '%s\n' % arg
            print(msg)
            raise type(e)(msg)
    return wrapper

def traceback_msg(func):
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            msg += 'args:\n'
            for arg in args:
                msg += '%s\n' % arg
            logging.error(msg)
            return msg
    return wrapper

def setup_logging(default_path='log_config.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG', overrides={}):
    """
    Setup logging configuration

    """
    logging.debug('%s called from %s.%s.' % (sys._getframe().f_code.co_name, sys._getframe(2).f_code.co_name,
                                             sys._getframe(1).f_code.co_name))
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            yaml_str = f.read()
        # logging.debug('yaml before overrides:\n' + yaml_str)
        for k, v in overrides.items():
            logging.debug('replacing %s with %s' % (k,v))
            yaml_str = yaml_str.replace(k,v)
        # logging.debug('yaml after overrides:\n' + yaml_str)
        config = yaml.safe_load(yaml_str)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return

def make_dir(dir_path):
    for i in range(5):
        if os.path.exists(dir_path): return
        try:
            os.makedirs(dir_path, 0755)
            # print "Making: %s" % dir_path
            logging.info( "Making: %s" % dir_path)
        except OSError:
            # print "Unable to make: %s" % dir_path
            logging.warning("Unable to make: %s" % dir_path)
            time.sleep(5)
    return


def del_dir(dir_path):
    for i in range(5):
        if not os.path.exists(dir_path): return
        try:
            shutil.rmtree(dir_path)
            # print "Deleting: %s" % dir_path
            logging.info("Deleting: %s" % dir_path)
        except:
            # print "Unable to delete: %s" % dir_path
            logging.warning("Unable to delete: %s" % dir_path)
            time.sleep(5)
    return


def del_file(file_path):
    for i in range(5):
        if not os.path.exists(file_path):
            logging.warning("File not found: %s" % file_path)
            return
        try:
            os.remove(file_path)
            logging.info("Deleting: %s" % file_path)
            if not os.path.exists(file_path):
                return
        except:
            logging.warning("Unable to delete: %s" % file_path)
            os.remove('\\\\?\\' + file_path)
            time.sleep(5)
    return

def copy_file(source_fp, dest_dp):
    try:
        shutil.copy(source_fp, dest_dp)
        logging.info('Copying %s to %s ...' % (source_fp, dest_dp))
    except:
        logging.warning('Unable to copy %s to %s !' % (source_fp, dest_dp))
        shutil.copy('\\\\?\\' + source_fp, '\\\\?\\' + dest_dp)
        logging.info('Copying %s to %s ...' % ('\\\\?\\' + source_fp, '\\\\?\\' + dest_dp))
    return

def read_fov_list(input_path, list_fn=None):
    if list_fn == None:
        list_fn = 'FOV_List.txt'
    fov_list_path = os.path.join(input_path, list_fn)
    fov_list = []
    if os.path.exists(fov_list_path):
        with open(fov_list_path, 'rb') as fl_f:
            while True:
                line = fl_f.readline()
                if not line: break
                fov = line.strip('\r\n')
                fov_list.append(fov)
    return fov_list

def get_position_call_counts(readLen, readDist):
    #print 'get_position_call_counts'
    tempDist = deepcopy(readDist)
    # print 'tempDist', tempDist
    call_counts = []
    running_count = 0
    if type(readDist) == list:
        #print 'processing read distribution as list'
        while readLen:
            if tempDist: running_count += tempDist.pop(0)
            call_counts.insert(0, running_count)
            readLen -= 1
    else:
        #print 'processing read distribution as dist'
        while readLen:
            if tempDist: running_count += tempDist.pop(readLen, 0)
            call_counts.insert(0, running_count)
            readLen -= 1
    #print 'call_counts', call_counts
    return call_counts