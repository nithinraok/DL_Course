
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def sigmoid(x):
    return (1/(1+np.exp(-x)));

def get_weights(cur_layer,next_layer):
    """ Returns random weights to start with. including last row (all ones) for bias
    where cur_layer is no of neurons for current layer and
    next_layer is no of neurons for next layer"""
    z=np.random.random((cur_layer-1,next_layer));
    b=np.ones((1,next_layer));
    w=np.vstack((z,b));
    return w;

def perceptron(input_x,weights):
    return sigmoid(np.dot(input_x,weights));

    


# In[3]:


inp_x=np.linspace(0,7,8);
act_m = 4;
y=4*inp_x;


# In[4]:


print(inp_x)


# In[ ]:


lr=0.1;
guess_m= 1;
num_x=len(inp_x);
num_of_iter = 100;

for i in np.arange(num_of_iter ):
    dCdM = (sum(abs(guess_m*inp_x - y)*guess_m));
    guess_m=guess_m - (lr/num_x)*dCdM;
    print("dCdM is %f new_guess m is %f "%(dCdM,guess_m))

