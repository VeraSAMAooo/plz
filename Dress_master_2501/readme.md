## model version explinations 
#### Version0
The very first terrible model, never officially tested, only several weight samples were left.
* checkpoint path -- > './checkpoint/v0'
#### Version1
The first group of models that worked, although not that well... These models holds basic settings of HR-VITON. No discirminator for tocg, no gan feat loss for generator, no type mask, etc...
* checkpoin path -- > './checkpoint/v1'
* configuration path (tocg) -- > './Configs/Config_condition_v1.py'
* configuration path (generator) -- > './Configs/Config_generator_v1.py'
#### Version2
There are four main modifications of this model:
* Train with balanced dataset (trousers and skirts): 'tocg_no_gan_loss', 'Config_condition_v1.py'
* Add discriminator for tocg: 'tocg_gan_loss', 'Config_condition_v2.py'
* Add type mask for tocg: 'tocg_typemask', 'Config_condition_v3.py'
* Add type mask for both tocg and discriminator: 'Config_generator_v2.py'
#### Skirt Only

