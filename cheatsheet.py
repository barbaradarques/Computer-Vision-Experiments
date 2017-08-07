# cnn.summary() # full description of the architecture


# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)

# 1/0 # breakpoint

# for i, layer in enumerate(cnn.layers): # prints the layers names (so you can know how to refer to each one in "get_layer")
#    print(i, layer.name)

# lst = [10, 2, 3, 9]
# s = sorted(lst)    
# [s.index(x) for x in lst]

# mylist=['a','b','c','d','e']
# myorder=[3,2,0,1,4]
# mylist = [ mylist[i] for i in myorder]
# print mylist


# preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3))