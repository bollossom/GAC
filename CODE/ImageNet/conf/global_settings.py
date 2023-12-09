from datetime import datetime

CHECKPOINT_PATH = '/userhome/QXR/checkpoint'

# total training epoches
EPOCH =250
# initial learning rate
# INIT_LR = 0.1

# time of we run the script
TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

# tensorboard log dir
LOG_DIR = 'runs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 30
