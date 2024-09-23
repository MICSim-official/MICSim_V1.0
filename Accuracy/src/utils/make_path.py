import configparser
import os
config = configparser.ConfigParser()
config.read(os.getenv('CONFIG'))
base      = config['Path']['log_dir']
organize  = config['Path']['organize']
tag       = config['Path']['tag']

def makepath_logdir():
    logdir = base
    for level in organize.split(','):
        section,key = level.split('_')
        logdir = os.path.join(logdir,key+"="+config[section][key])
    logdir = os.path.join(logdir, 'tag' + "=" + tag)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

