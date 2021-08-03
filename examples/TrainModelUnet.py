#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


BaseDir = '/home/sancere/Kepler/CurieTrainingDatasets/Dalmiro_Laura/'
NPZdata = 'WingVeinModelUNET.npz'

ModelDir ='/home/sancere/Kepler/CurieDeepLearningModels/Dalmiro_Laura/'
ModelName = 'WingVeinUNET'
load_path = BaseDir + NPZdata 


# In[3]:


(X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.05, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


# In[4]:


plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');


# In[5]:


config = config = Config(axes, n_channel_in, n_channel_out, probabilistic = False, unet_n_depth=5,unet_n_first = 48,unet_kern_size = 7, train_loss = 'mae', train_epochs= 150, train_learning_rate = 1.0E-4 ,train_batch_size = 1,  train_reduce_lr={'patience': 5, 'factor': 0.5})
print(config)
vars(config)


# In[6]:


model = CARE(config = config, name = ModelName, basedir = ModelDir)
#input_weights = ModelDir + ModelName + '/' +'weights_best.h5'
#model.load_weights(input_weights)


# In[ ]:


history = model.train(X,Y, validation_data=(X_val,Y_val))


# In[ ]:


print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);


# In[ ]:


plt.figure(figsize=(12,7))
_P = model.keras_model.predict(X_val[25:30])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[0],Y_val[0],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');


# In[ ]:





# In[ ]:




