import monkdata as m
import dtree as dtree
import numpy as np
import dtree as d
#import drawtree_qt4 as dt
import math




t = d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1test))

"Entropy Calculation"
# test for entropy caluclation
# false records
v= 0
for i in range(len(m.monk2)):
    if m.monk2[i].positive==False:
        v +=1
        #print('i: ',(v,m.monk2[i].positive))
    #else:
        #print('no')
        

#unlabeled data is more predictable if entropy is low
def entropy(dataset):
    "Calculate the entropy of a dataset"
    #nr of monk1 records
    n = len(dataset)
    # nr of monk1 records with postive = True
    nPos = len([x for x in dataset if x.positive])
    #nr of monk1 records with positive = False
    nNeg = n - nPos
    #if all records are negative or all are positive than entropy is 0 since one can immediately classify or predict unlabeled records.
    if nPos == 0 or nNeg == 0:
        return 0.0
    #Entropy calc
    return -float(nPos)/n * math.log(float(nPos)/n,2) + \
        -float(nNeg)/n * math.log(float(nNeg)/n,2)

print('Monk1 entropy:',dtree.entropy(m.monk1))
print('Monk2 entropy:',dtree.entropy(m.monk2))
print('Monk3 entropy:',dtree.entropy(m.monk3))

"Entropy Calculation"

"Information Gain calculation"

def averageGain(dataset, attribute):
    "Calculate the expected information gain when an attribute becomes known"
    weighted = 0.0
    #ex monk1: A1 attribute values are {1,2,3} v= 1 or 2 or 3
    for v in attribute.values:
        #ex monk1: for v=1 subset = (True,  (1, 1, 1, 1, 3, 1), 5) , v=2 subset= (False, (2, 1, 1, 1, 3, 1), 149), v=3 subset = (True,  (3, 1, 1, 1, 1, 1), 289)
        #select: selects all samples with attribute= v
        subset = select(dataset, attribute, v)
        # entropy of subset is how predictable is an unlabeled record to be positive or negative and has attribute 1
        # weighted is the sum of entropies for attribute 1
        weighted += entropy(subset) * len(subset)
    # entropy(monk1) - entropy(subset)
    # the less entropy of subset is, the better it's at classifiying data. So, the higher the information gain is the better it's to use this attribute to split.
    return entropy(dataset) - weighted/len(dataset)


def select(dataset, attribute, value):
    "Return subset of data samples where the attribute has the given value"
    return [x for x in dataset if x.attribute[attribute] == value]

#test for how select function works 
for v in m.attributes[0].values:
    subset = dtree.select(m.monk1, m.attributes[0],v)


#information Gain calculation

Data= np.array([np.array(m.monk1),np.array(m.monk2), np.array(m.monk3)])
'''
Dictionary where keys are the IG values and values are the attribute names 
'''
IG = {}
for i in range(3):
    for j in range(len(m.attributes)):
        IG[dtree.averageGain(Data[i], m.attributes[j])] = m.attributes[j].name
    #max IG
    maxIGKey = max(IG.keys())
    #the chosen attribute that has the highest value of IG such as A5 in monk1
    maxIGValue = IG[maxIGKey]
    print('chosen root node for monk%d: %s'%(i,maxIGValue))


"Information Gain calculation"


"Building Decision Trees"


def chosenNode(DataSet, attributes):
    IG = {}
    for j in range(len(attributes)):
        IG[dtree.averageGain(DataSet, attributes[j])] = attributes[j].name
    #max IG
    maxIGKey = max(IG.keys())
    #the chosen attribute that has the highest value of IG such as A5 in monk1
    maxIGValue = IG[maxIGKey]
    #print('chosen node: %s'%(maxIGValue))
    return maxIGValue

DataSet = m.monk1
#original attributes
attributes = m.attributes
# attr left after choosing node
attr = {}
chosenAttr = []
#subsets of a5
subset= []

#dictionary of attribute names and values
for i in range(len(m.attributes)):
    attr[m.attributes[i].name]= m.attributes[i].values
    if m.attributes[i].name == chosenNode(DataSet, m.attributes):
        chosenAttr = attributes[i]
        #remove chosen attribute from attr dictionary
        
        
print('attributes',type(attributes[0]))
print(attr.get(chosenNode(DataSet, m.attributes)))

# generating subsets for different values in a5 attribute
for i in range(len(attr.get(chosenNode(DataSet, m.attributes)))):
    subset.append(select(DataSet, chosenAttr, attr.get(chosenNode(DataSet, m.attributes))[i]))

#finds the first level of chosen attribute as a node
for i in range(len(subset)):
    for j in range(len(attributes)):
        nextChosen = chosenNode(subset[i], attributes)
        print('info gain for subset[%d] and a%d: %s'%(i,j, nextChosen))
        #print('subset',subset[i][j].identity)






"Building Decision Trees"