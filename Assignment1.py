#!/usr/bin/env python
# coding: utf-8

# # Muhammad Anas Hassan

# Roll NO: PIAIC121089 

# 
# # Assignment For Numpy

# # Difficulty level Begineer

# 1. Import the numpy package under the name np

# In[7]:


import numpy as np


# 1. Create a null vector of size 10

# In[9]:


c=np.zeros(10)
c


# 3.Create a vector with values ranging from 10 to 49

# In[11]:


c=np.arange(10,50)
c


# 4. Find the shape of previous array in question 3

# In[12]:


c.shape


# 5. Print the type of the previous array in question 3

# In[13]:


print(type(c))
c.dtype


# 6. Print the numpy version and the configuration

# In[14]:


print('numpy version is',np.__version__)
np.show_config()


# 7. Print the dimension of the array in question 3

# In[15]:


c.ndim


# 8. Create a boolean array with all the True values

# In[16]:


bool=c>0
bool 


# 9. Create a two dimensional array

# In[17]:


b2= np.array([[0,1,3],[3,1,0]])
print(b2, '\n')
print('verification',b2.ndim)


# 10. create a three dimensional array

# In[18]:


b3= np.array([[[0,1,3],[3,1,0]]])
print(b3, '\n')
print('verification',b3.ndim)


# Difficulty level Easy

# 1. Reverse a vector (first element becomes last)

# In[16]:


import numpy as  np


# In[17]:


vec = np.arange(10,50,10)
print('original vector',vec,'\n\n')
rev_vec = vec[::-1]
print('Reverse vector',rev_vec)


# 2. Create a null vector of size 10 but the fifth value which is 1

# In[18]:


null_vector=np.zeros(10)
print(null_vector,'\n\n')
null_vector[4] = 1
print(null_vector)


# 3. Create a 3x3 identity matrix

# In[19]:


identity_matrix = np.eye(3)
print(identity_matrix)


# 4. arr = np.array([1, 2, 3, 4, 5])
# 
# 
# Convert the data type of the given array from int to float

# In[20]:


arr = np.array([1,2,3,4,5])
print(arr.dtype)


f_arr = np.array([1,2,3,4,5], dtype=float)
print(f_arr.dtype)


# 1.arr1= np.array([[1.,2.,3.]
#                       
#                       [4.,5.,6.]]
#   arr2= np.array([[0.,4.,1.]
#                       
#                       [7.,2.,12.]]
# Multiply arr1 with arr2

# In[21]:


arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[0,4,1],[7,2,12]])
print(arr1*arr2)


# 1.arr1= np.array([[1.,2.,3.]
# 
#                       [4.,5.,6.]]
#  
#   arr2= np.array([[0.,4.,1.]
#   
#                       [7.,2.,12.]]
# Make an array by comparing both the arrays provided above

# In[24]:


arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[0,4,1],[7,2,12]])
(np.equal(arr1,arr2))


# 1. Extract all odd numbers from arr with values(0-9)

# In[26]:


num = np.arange(10)
print(num)


print(num[num % 2 == 1])


# 1. Replace all odd numbers to -1 from previous array

# In[34]:


a = num = np.arange(10)
print(a, '\n')

num[num % 2 == 1] = -1

print(num)


# 1. arr = np.arange(10)
# 
# 
# 
# Replace the values of indexes 5,6,7 and 8 to 12

# In[35]:


arr = np.arange(10)
print(arr,'\n')
arr[5:9]=12

print(arr)


# 1. Create a 2d array with 1 on the border and 0 inside

# In[36]:


ar2 = np.ones((5,5), dtype=int)
print(ar2,'\n')

ar2[1:-1,1:-1] = 0
print(ar2)


# Difficulty level Medium

# 1. arr2d = np.array([[1,2,3]
# 
#                        [4,5,6]
#                        
#                        [7,8,9]
# 
# 
# Replace the value 5 to 12

# In[38]:


arr2d = ([[1,2,3],[4,5,6],[7,8,9]])
arr2d[1][1]=12
print(arr2d,'\n')


# 1. arr3d = ([[[1,2,3],[4,5,6]],[[7,8,9][10,11,12]]])
# 
# 
# Convert all the values of 1st array to 64

# In[39]:


arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr3d,'\n\n\n')

arr3d[:1,0:2] = 64
print(arr3d)


# 1. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
# 

# In[42]:


Array2D = np.arange(9).reshape(3,3)
print(Array2D,'\n\n\n')

Array1D = Array2D[0]
print(Array1D)


# 1. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[43]:


Array2d = np.arange(9).reshape(3,3)
print(Array2D,'\n\n\n')

V2=Array2D[1][1]
print("2nd value from 2nd 1D array is:",V2)


# 1. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[44]:


Array2D = np.arange(9).reshape(3,3)
print(Array2D,'\n\n\n')

C3=Array2D[:2,2:]
print(C3)


# 1. Create a 10x10 array with random values and find the minimum and maximum values

# In[46]:


rand_array = np.random.random((8,8))
print(rand_array,'\n\n\n')
print("max = ",rand_array.max(),"\n\nmin = ",rand_array.min())


# # 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])

# 1. find the common item between a and b

# In[47]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("commom item between a and b are",np.intersect1d(a,b))


# 1. Find the positions where elements of a and b match

# In[48]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# 1. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# 
# Find all the values from array data where the values from array names are not equal to Will

# In[4]:


import numpy as np


# In[5]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)

data[names != "Will"]


# 1. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# 
# 
# Find all the values from array data where the values from array names are not equal to Will and Joe

# In[6]:


names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(7,4)
data


# In[7]:


print(names == "Will")
print(names == "Joe")

L = (names == "Will") | (names == "Joe")
print(names[L])
print(data[L])

print('\n',names[~ L])
print(data[~L])


# Difficulty level Hard

# 1. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[11]:


Arr2d = np.arange(1,16).reshape(5,3)
Arr2d


# 1. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[12]:


u = np.arange(1,17).reshape(2,2,4)
print(u)
print("\n\nshape:",u.shape)


# 1. Swap axes of the array you created in Question 32

# In[16]:


np.swapaxes(z,0,1)


# 1. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[17]:


a = np.arange(10)
print(a, '\n\n')

print(np.sqrt(a))


# 1. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays 

# In[19]:


M = np.random.randn(12)
print("M= " ,M)

N = np.random.randn(12)
print("\nN = " ,N)

O=np.maximum(M,N)
print('\n maximum values array',O)


# 1. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# 
# Find the unique names and sort them out!

# In[20]:


Names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
u_a_s=np.unique(np.sort(Names))
print(u_a_s)


# 1. a = np.array([1,2,3,4,5]) b = np.array([5,6,7,8,9])
# 
# 
# From array a remove all items present in array b

# In[21]:


a = np.array([1,2,3,4,5])
print(a, '\n')
b = np.array([5,6,7,8,9])
print(b, '\n')
a = np.setdiff1d(a,b)
print("after removing a becomes", a)


# 1. Following is the input NumPy array delete column two and insert following new column in its place.
# 
# 
# 
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])
# 
# 
# 
# 
# newColumn = numpy.array([[10,10,10]])

# In[23]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = np.array([[10,10,10]])
print(sampleArray, '\n\n')

sampleArray=np.delete(sampleArray,np.s_[1:2],axis=1)
sampleArray=np.insert(sampleArray,1,newColumn,axis=1)

print(sampleArray)


# 1. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# 
# 
# Find the dot product of the above two matrix 

# In[24]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(x,'\n\n',y)
print('\n',np.dot(x,y))


# 1. Generate a matrix of 20 random values and find its cumulative sum

# In[25]:


C=np.random.randn(20)
C.cumsum()

