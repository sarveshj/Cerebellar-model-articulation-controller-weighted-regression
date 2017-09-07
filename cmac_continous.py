#!/usr/bin/env python3


import math
import numpy as np
from collections import defaultdict
import sys




hash_mapping=defaultdict(dict)


									#-----------------------------
									# Task 1:generate input sample
									#------------------------------


data=np.linspace(0,0.5*np.pi,100)

#generate train and test!
input_train=[data[x] for x in range(0,len(data)) if x%3!=0]
input_test=[data[y] for y in range(0,len(data)) if y%3 ==0]

#subsample evry 3 samples train to generate validation set
input_val=input_train[0:len(input_train):3]

#sanity check!
if len(set(input_val).intersection(set(input_test)))!=0 or len(set(input_val).intersection(set(input_train)))!=0 and len(set(input_train).intersection(set(input_test)))!=0 :
	print("No cheating ..there is overlap!!")

												#-------------------------------------#
												# Task 2: Scale the data & quantize it
												#-------------------------------------#



target_train=np.cos(input_train)
target_val=np.cos(input_val)
target_test=np.cos(input_test)



train_min=min(target_train)
train_max=max(target_train)
train_range=train_max-train_min


association_function=1000 # used to map input to weights


#gen?
gen=int(input("Enter gen value"))

#print(target_train[:10])
#sys.exit(1)


"""
#min-max normalized if only we have non-normalized curve/ negative values
for val in target_train.flatten():
	#modified min-max normalization for negative numbers!
	val=(val-abs(train_min))/train_range
	target_train_mod.append(val)	
"""



												#-----------------------#
												# Generate Hash Mapping #
												#-----------------------#

#map from input values --> 
input_levels=set(input_train)
for i in input_levels:
	#convert numpy.float to int..
	i=round(float(i*association_function),4)
	#one extra for continous
	for j in range(int(i),int(i)+gen+1):
		#print(i,j)
		hash_mapping[i][j]=0
#print(hash_mapping)	
#sys.exit(1)

lookup_table=dict()
count=0
for i in input_levels:
	#convert numpy.float to int..
	i=round(float(i*association_function),4)
	lookup_table[i]=target_train[count] # index 0: o--> cos(0)
	count+=1

#round off target values for continous CMAC
#target_train=[round(float(i*association_function),4) for i in target_train]
#target_val=[round(float(i*association_function),4) for i in target_train]
#target_test=[round(float(i*association_function),4) for i in target_test]



										#---------------------------------------------------------#
										# Calculate error, update weights and update weight vector
										#---------------------------------------------------------#
	
epoch=1
epoch_count= 100
error_list=list()
n_f=0

while epoch <=epoch_count:
	error_sum=0
	count=0
	for i in input_train:

		map_value=round(float(i*association_function),4)

		#initialize the weight sum to zero ..
		# read prev weight value and compare 
		#with taget
		if map_value not in hash_mapping.keys():
			print("not found")
			n_f+=1
			continue

		if map_value  in hash_mapping.keys():
			
			weight_sum=0
			for v in hash_mapping[map_value].values():
				#print("k---> %f v---> %f"%(k,v))
				weight_sum=weight_sum+v

			""" Error calculation """
			# y(estimate) -target
			error=target_train[count]-weight_sum
			error_delta=error/gen
			error_sum+=error


			#print("error delta is %f"%(error_delta))
			#error_list.append(error_delta)
			#sys.exit(1)

			beta=1#learning!!	
			if error >= 0:

				for idx,k in enumerate(hash_mapping[map_value].keys()):
					if idx==0:  
						#1st weight
						hash_mapping[map_value][k]+=0.5*error_delta
					elif idx==gen:
						hash_mapping[map_value][k]+=0.5*error_delta
					else:
						hash_mapping[map_value][k]+=error_delta

			"""
			#if I include this ... error jumps!!	
			elif error < 0:
				for k in hash_mapping[map_value].keys():
					hash_mapping[map_value][k]-=error_delta
			else:
				pass

			"""
		"""
		if map_value not in hash_mapping.keys():
			print("match value is not found %f"%(map_value))
			n_f+=1
		"""
		count+=1
	error_sum =error_sum/len(target_train)
	error_list.append(error_sum)		
	print("error delta %f, error sum for  %d epoch is %f "%(error_delta,epoch,error_sum))
	epoch+=1
#sys.exit(1)

def find_closest_key(map_value,hash_mapping):
	t=[abs(map_value-x) for x in hash_mapping.keys()]
	#print("mapped value for %f ---> hash_mapping[%f]"%(map_value, np.argmin(t)))
	return t.index(min(t))


val_error=0
validation="False"
if validation=="True":
	for idx,v in enumerate(input_val):
		map_value=round(float(v*association_function),4)
		#print(map_value)
		#initialize the weight sum to zero ..
		if map_value not in hash_mapping.keys():
			#print("validation map value not found ... estimating nearest one")
			#continue
			k=find_closest_key(map_value,hash_mapping)
			new_map_value=list(hash_mapping.keys())[k]
			map_value=new_map_value
			#print("new --> %f old --> %f"%(new_map_value, map_value))


		#if map_value  in hash_mapping.keys():
			#initialize weight sum for each validation sample
		weight_sum=0 
		for l in hash_mapping[map_value].values():
			#print("k---> %f "%(k,v))
			weight_sum=weight_sum+l
		#print("v --> %f actual --> %f predicted val --> %f"%(v, target_val[idx],weight_sum))
		val_error+=target_val[idx]-weight_sum


	print("Average validation error for gen value %d is---> %.6E"%(gen,val_error/len(input_val)))
	

#sys.exit(1)

#testing
test_error=0
predicted_list=list()

for idx,t in enumerate(input_test):

	map_value=round(float(t*association_function),4)
	#print(map_value)
	#initialize the weight sum to zero ..
	if map_value not in hash_mapping.keys():
		k=find_closest_key(map_value,hash_mapping)
		new_map_value=list(hash_mapping.keys())[k]
		map_value=new_map_value
	
	weight_sum=0 
	for l in hash_mapping[map_value].values():
		#print("k---> %f v---> %f"%(k,v))
		weight_sum=weight_sum+l
	#print("t --> %f actual --> %f predicted val --> %f"%(t, target_test[idx],weight_sum))
	test_error+=target_test[idx]-weight_sum
	predicted_list.append(weight_sum)
print("Average test error for gen value %d is---> %.6E"%(gen,test_error/len(input_test)))


#plot results

from matplotlib import pyplot as plt
plt.plot(input_test,predicted_list,color="blue",linewidth="2.5",linestyle="-")
plt.plot(input_test,target_test,'r*',markersize=12)
plt.title('Plot of Original and Apprximated Function using Continous CMAC')
plt.plot(input_test,predicted_list,color="blue",linewidth="2.5",linestyle="-")
plt.plot(data,np.cos(data),'r*',markersize=12)
plt.legend(['Approximated Function','Original Function'])
plt.grid()
plt.show()
