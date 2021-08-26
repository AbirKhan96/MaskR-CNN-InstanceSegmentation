#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
root_folder = "folder/"
for folder in  os.listdir(root_folder) :
    print (folder)
    for fname in os.listdir(os.path.join(root_folder, folder)):
        print ("         ---->", fname)
        os.rename(os.path.join(root_folder, folder, fname), os.path.join(root_folder, folder, folder+'_'+fname))


# In[ ]:




