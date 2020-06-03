
import csv
import math
import operator

def euclidean_distance(instance1, instance2, length):
    distance=0
    for x in range(length):
        distance+=(instance1[x]-instance2[x])**2
    return math.sqrt(distance)

def get_neighbors(trainingset, test_instance, k):
    distances=[]
    length=len(test_instance)-1
    
    for x in range(len(trainingset)):
        dist=euclidean_distance(test_instance, trainingset[x], length)
        distances.append((trainingset[x],dist))
    
    distances.sort(key=operator.itemgetter(1))
    
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors

def get_response(neighbors):
    a=0
    b=0
    
    for x in range(len(neighbors)):
        response=neighbors[x][-1]

        if response == 0.0:
            a+=1
        elif response == 1.0:
            b+=1
    
    if a > b:
       return 0.0
    else:
       return 1.0
    
    #sorted_votes=sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)
    


def get_accuracy(testset, predictions):
    correct=0

    for x in range(len(test_set)):
        if testset[x][-1] == predictions[x]:
            correct+=1
        else:
            pass
    return (correct/float(len(testset)))*100.0


training_set=[]
test_set=[]
    
with open('spam_train.csv', 'r') as train_csvfile:
    lines=csv.reader(train_csvfile)
    next(lines)
    dataset=list(lines)
    
    for i in range(len(dataset)):
         for j in range(58):
             dataset[i][j]=float(dataset[i][j])
         training_set.append(dataset[i])
    
with open('spam_test.csv', 'r') as test_csvfile:
    lines=csv.reader(test_csvfile)
    next(lines)
    dataset=list(lines)
    
    for i in range(len(dataset)):
        dataset[i].pop(0)
        for j in range(58):
            dataset[i][j]=float(dataset[i][j])
        test_set.append(dataset[i])
    
k=[1,5,11,21,41,61,81,101,201,401]

for k_num in k:
    predictions=[]
    for i in range(len(test_set)):
        neighbors=get_neighbors(training_set, test_set[i],k_num)
        result=get_response(neighbors)
        predictions.append(result)
   
    accuracy=get_accuracy(test_set, predictions)
 
    print('For k = {}, the test accuracy is: {}%.'.format(k_num, accuracy) )
         
