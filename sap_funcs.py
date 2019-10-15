import logging.config
logger = logging.getLogger(__name__)
import os
import shutil
import yaml
import sys
import time
from copy import deepcopy

int_extensions = {
    'v1': 'bin',
    'v2': 'int',
    'BB': 'csv'
}

def chmod_paths(path_list):
    for path in path_list:
        os.chmod(path, )

def generate_service_tag():
    import random, string
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))

def prepare_json_dict(parameter_overrides):
    import json
    if type(parameter_overrides) == dict:
        parameters = parameter_overrides
    else:
        parameters = json.load(open(parameter_overrides, 'r'))
    # Remove blank value entries
    for k, v in parameters.items():
        if not v:
            logger.debug('Removing empty key: %s' % k)
            parameters.pop(k)
    return parameters

def log_tag(instance, log_str):
    """
    Wrapper for log statements to include relevant information in an easily modifiable manner

    :param instance:
        class instance that contains the relevant information (fov, lane, etc.)
    :param log_str:
        original log string
    :return:
    """
    return '%s - %s' % (instance.fov, log_str)

def output_table(output_fp, list_table, header=[], delimiter=','):
    import csv
    if header:
        list_table = [header] + list_table
    with open(output_fp, 'wb') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(list_table)
    return


def output_df(output_fp, data, index_list):
    import pandas as pd
    df = pd.DataFrame(data, index=index_list)
    df = df.dropna()
    df.to_csv(output_fp)
    return


def get_magic_fovs():
    return ['C%03dR%03d' % (col, row) for col in [2, 7] for row in [8, 23, 38]]

def sendemail(sender_id, email_list, subject, text, files=None,
              server="lxv-intmx01.gc01.cgiprod.com:25"):
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate
    import smtplib

    sender_address = '%s@completegenomics.com' % sender_id
    assert isinstance(email_list, list)

    msg = MIMEMultipart()
    msg['From'] = sender_address
    msg['To'] = COMMASPACE.join(email_list)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        if len(f) > 260: f = '\\\\?\\' + f
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=os.path.basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
        msg.attach(part)


    smtp = smtplib.SMTP(server)
    problems = smtp.sendmail(sender_address, email_list, msg.as_string())
    smtp.close()
    return

def full_traceback(func):
    """
    Wraps function to provide traceback on exception.
    Will raise exception, halting execution.
    This is the original function. Its use has deprecated as it has been upgraded with email functionality.
    :param func:
    :return:
    """
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            msg += 'args:\n'
            for arg in args:
                if type(arg) == dict:
                    msg += '{\n'
                    for k, v in arg.items():
                        msg += '%s: %s\n' % (k, v)
                    msg += '}\n'
                msg += '%s\n' % arg
            raise type(e)(msg)
    return wrapper

def traceback_msg(func):
    """
    Wraps function to provide traceback on exception.
    Will return exception in string format for logging purposes.
    This brand of traceback is used for subprocesses.
    :param func:
    :return:
    """
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            msg += 'args:\n'
            for arg in args:
                if type(arg) == dict:
                    msg += '{\n'
                    for k, v in arg.items():
                        msg += '%s: %s\n' % (k, v)
                    msg += '}\n'
                else:
                    msg += '%s\n' % arg
            logger.error('%s' % msg)
            return msg
    return wrapper

def traceback_hold(func, hostname, emails):
    import traceback, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            sbj = 'ERROR! %s' % str(e).upper()
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            msg += '\n%s placed on temporary hold. Please log on to resume function.' % hostname
            sendemail(hostname, emails, sbj, msg)
            logger.error(sbj)
            logger.error(msg)
            raw_input('Press any key to exit.')
    return wrapper

def setup_logging(config_path='log_config.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG', overrides={}):
    """
    Setup logging configuration

    """
    logger.debug('%s called from %s.%s.' % (sys._getframe().f_code.co_name, sys._getframe(2).f_code.co_name, sys._getframe(1).f_code.co_name))
    path = config_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            yaml_str = f.read()
        #logger.debug('yaml before overrides:\n' + yaml_str)
        for k, v in overrides.items():
            logger.debug('replacing %s with %s' % (k,v))
            yaml_str = yaml_str.replace(k,v)
        #logger.debug('yaml after overrides:\n' + yaml_str)
        config = yaml.safe_load(yaml_str)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return

def make_dir(dir_path):
    for i in range(5):
        if os.path.exists(dir_path): return
        try:
            os.makedirs(dir_path, 0o755)
            #print "Making: %s" % dir_path
            logger.info( "Making: %s" % dir_path)
        except OSError:
            #print "Unable to make: %s" % dir_path
            logger.warning("Unable to make: %s" % dir_path)
            time.sleep(5)
    return

def del_dir(dir_path):
    for i in range(12):
        if not os.path.exists(dir_path): return
        try:
            shutil.rmtree(dir_path)
            #print "Deleting: %s" % dir_path
            logger.info("Deleting: %s" % dir_path)
        except Exception as e:
            #print "Unable to delete: %s" % dir_path
            #logger.warning("Unable to delete: %s" % dir_path)
            logger.warning("shutil.rmtree exception!")
            logger.warning(str(e))
            time.sleep(i*5)
    return

def del_file(file_path):
    for i in range(5):
        if not os.path.exists(file_path):
            logger.warning("File not found: %s" % file_path)
            return
        try:
            os.remove(file_path)
            logger.info("Deleting: %s" % file_path)
            if not os.path.exists(file_path):
                return
        except:
            logger.warning("Unable to delete: %s" % file_path)
            os.remove('\\\\?\\' + file_path)
            time.sleep(5)
    return

def copy_file(source_fp, dest_dp):
    try:
        shutil.copy(source_fp, dest_dp)
        logger.info('Copying %s to %s ...' % (source_fp, dest_dp))
    except:
        logger.warning('Unable to copy %s to %s !' % (source_fp, dest_dp))
        shutil.copy('\\\\?\\' + source_fp, '\\\\?\\' + dest_dp)
        logger.info('Copying %s to %s ...' % ('\\\\?\\' + source_fp, '\\\\?\\' + dest_dp))
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
    print('tempDist: {0}'.format(tempDist))
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