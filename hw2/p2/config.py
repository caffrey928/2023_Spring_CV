################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name   = 'sgd_pre_da' # name of experiment

# Model Options
model_type = 'resnet18' # 'mynet' or 'resnet18'
# model_type = 'mynet'

# Learning Options
epochs     = 50           # train how many epochs
batch_size = 32           # batch size for dataloader, original = 32
use_adam   = False        # Adam or SGD optimizer
lr         = 1e-2         # learning rate, original = 1e-2
milestones = [16, 32, 45] # reduce learning rate at 'milestones' epochs
