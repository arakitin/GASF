# Introduction to machine learning
# Excercise 1.5
# Matti Vuorinen 6.11.2009
#
# A simple and inefficient kNN-implementation, no input error checking.
# Usage python kNN train_file test_file k
# Enable psyco if you have it installed. It gives about 10x speed boost.

import sys
import random
import matplotlib.pyplot as plt
import numpy as np

import math

#import psyco
#psyco.full()
#from kmeans import kmeans_euclid,kmeans_hamming,kmeans_my_hamming

def convMatrix( label, rsids, fdData, people):
    convMatrix = []
    for count in range(len(fdData[0])):
        if (fdData[0][count] in people):
            convMatrix.append([label])
    header = True
    for line in fdData:
        if header:
            header = False
            continue # next line
        c = 0
        if (line[0] in rsids):
            for count in range(len(line)):
                if (fdData[0][count] in people):
                    convMatrix[c].append( line[count])
                    c = c + 1 
    return convMatrix

def loadMatrix( fdData):
    dataMatrix = []
    line_count = 0
    for line in fdData:
        if line_count == 10001:
            break
        rsid = []
        good = True
        count = 0
        for token in line.split():
            count = count + 1
            if token == 'NN':
                good = False
                break
            if 1 < count < 12:
                continue
            rsid.append( token)
        if good:
            line_count = line_count + 1
            dataMatrix.append( rsid)
    return dataMatrix

#Computes the Hamming distance
def getDistance( sample1, sample2):
    distance = 0
    for i in range( 0, len( sample2)):
        if len(sample1) != len(sample2):
            print 'sample1',len(sample1),'sample2',len(sample2)
        if not sample1[i] == sample2[i]:
            distance = distance + 1
    return distance

def getNearestNeighbours( sample, dataMatrix, k):
    distances = []
    for i in range( 0, k):
        distances.append( sys.maxint)
    neighbours = []

    for i in range( 0, len( dataMatrix)):
        neighbour = dataMatrix[ i]
                        
        distance = getDistance( sample[1:], neighbour[1:]) #Drop the class
        for j in range( 0, k):
            if distance < distances[ j]:
                distances = distances[:j] + [distance] + distances[j:-1]
                neighbours = neighbours[:j] + [neighbour] + neighbours[j:-1]
                break

    return neighbours

def majorityVote( samples):
    votes = {}
    for sample in samples:
        if sample[0] in votes:
            votes[ sample[0]] = votes[ sample[0]] + 1
        else:
            votes[ sample[0]] = 1
    
    max = 0
    maxVote = None
    for vote in votes.items():
        if vote[ 1] > max:
            max = vote[ 1]
            maxVote = vote[0]

    return maxVote


def main():
    fpCEU = open("genotypes_chr11_CEU.b35.txt", "r")
    CEU = loadMatrix(fpCEU)
    fpCEU.close()

    fpCHB = open("genotypes_chr11_CHB.b35.txt", "r")
    CHB = loadMatrix(fpCHB)
    fpCHB.close()

    fpJPT = open("genotypes_chr11_JPT.b35.txt", "r")
    JPT = loadMatrix(fpJPT)
    fpJPT.close()

    fpYRI = open("genotypes_chr11_YRI.b35.txt", "r")
    YRI = loadMatrix(fpYRI)
    fpYRI.close()

    rsids_CEU = []
    for rec in CEU:
        if rec[0] != 'rsid':
            rsids_CEU.append(rec[0])

    rsids_CHB = []
    for rec in CHB:
        if rec[0] != 'rsid':
            rsids_CHB.append(rec[0])

    rsids_JPT = []
    for rec in JPT:
        if rec[0] != 'rsid':
            rsids_JPT.append(rec[0])

    rsids_YRI = []
    for rec in YRI:
        if rec[0] != 'rsid':
            rsids_YRI.append(rec[0])

    rsids = list(set(rsids_CEU) & set(rsids_CHB) & set(rsids_JPT) & set(rsids_YRI))

    print 'rsids',len(rsids)

    people = CEU[0][1:] + CHB[0][1:] + JPT[0][1:] + YRI[0][1:]

    print 'people',len(people)
    print 'people',len(set(people))

    train_people = []
    test_people = []
    for index in range(len(people)):
        if index%2==0:
            train_people.append(people[index])
        else:
            test_people.append(people[index])

    print 'train',len(train_people)
    print 'test',len(test_people)

    trainMatrixCEU = convMatrix( 'CEU', rsids, CEU, train_people)
    trainMatrixCHB = convMatrix( 'CHB', rsids, CHB, train_people)
    trainMatrixJPT = convMatrix( 'CHB', rsids, JPT, train_people)
    trainMatrixYRI = convMatrix( 'YRI', rsids, YRI, train_people)

    trainMatrix = []
    for rec in trainMatrixCEU:
        trainMatrix.append(rec)
    for rec in trainMatrixCHB:
        trainMatrix.append(rec)
    for rec in trainMatrixJPT:
        trainMatrix.append(rec)
    for rec in trainMatrixYRI:
        trainMatrix.append(rec)

    testMatrixCEU = convMatrix( 'CEU', rsids, CEU, test_people)
    testMatrixCHB = convMatrix( 'CHB', rsids, CHB, test_people)
    testMatrixJPT = convMatrix( 'CHB', rsids, JPT, test_people)
    testMatrixYRI = convMatrix( 'YRI', rsids, YRI, test_people)

    testMatrix = []
    for rec in testMatrixCEU:
        testMatrix.append(rec)
    for rec in testMatrixCHB:
        testMatrix.append(rec)
    for rec in testMatrixJPT:
        testMatrix.append(rec)
    for rec in testMatrixYRI:
        testMatrix.append(rec)

    res = []
    for k in range(1, 101):
        res.append(doOneK(trainMatrix, testMatrix, k))

    plt.figure()
    plt.plot(np.arange(1,101), res)
    plt.show()

    dataCEU = convMatrix( 'CEU', rsids, CEU, people)
    dataCHB = convMatrix( 'CHB', rsids, CHB, people)
    dataJPT = convMatrix( 'JPT', rsids, JPT, people)
    dataYRI = convMatrix( 'YRI', rsids, YRI, people)

    data = []
    for rec in dataCEU:
        data.append(rec)
    for rec in dataCHB:
        data.append(rec)
    for rec in dataJPT:
        data.append(rec)
    for rec in dataYRI:
        data.append(rec)

    init_centroids = []
    init_centroids.append(dataCEU[0])
    init_centroids.append(dataCHB[0])
    #init_centroids.append(dataJPT[0])
    init_centroids.append(dataYRI[0])
    kmeans_my_hamming(data,init_centroids,0)

def my_hamming_dist(x,y):
    	return sum(x[i]!=y[i] for i in range(len(x)))

def min_index(a):
    amin = a[0]
    imin = 0
    for i in range(1,len(a)):
        if a[i]<amin:
            amin = a[i]
            imin = i
    return imin

def max_count(count):
    mx = 0
    mm = None
    for q in count.keys():
        if count[q] > mx:
            mx = count[q]
            mm = q
    return mm

def make_centroids(cl):
    ret = []
    for c in range(len(cl)):
        centroid = []
        for p in range(len(cl[c][0])):
            count = {}
            for d in cl[c]:
                if d[p] not in count.keys():
                    count[d[p]] = 1
                else:
                    count[d[p]] += 1
            centroid.append(max_count(count))
        ret.append(centroid)
    return ret

def max_diff(x,y):
    ret = 0
    for i in range(len(x)):
        tmp = my_hamming_dist(x[i][1:],y[i][1:])
        if tmp > ret:
            ret = tmp
    return ret
#    return abs(max(my_hamming_dist(x[i],y[i]) for i in range(len(x))))

def kmeans_my_hamming(data,centroids,cutoff):
    cl = {}
    while True:
        for c in range(len(centroids)):
            cl[c] = []
        for d in range(len(data)):
            dist_to_centroids = []
            for c in range(len(centroids)):
                dist_to_centroids.append(my_hamming_dist(centroids[c][1:], data[d][1:]))
            index = min_index(dist_to_centroids)
            cl[index].append(data[d])

        new_centroids = make_centroids(cl)
        if max_diff(centroids, new_centroids) <= cutoff:
            break
        
        centroids = new_centroids

    s = 0
    for d in data:
        averDiss = {}
        curr = None
        for c in cl.keys():
            averDiss[c] = 0
            for r in cl[c]:
                dist = my_hamming_dist(r[1:],d[1:])
                if dist==0:
                    curr = c
                averDiss[c] += dist/1.0/len(cl[c])
        a = averDiss[curr]
        averDiss[curr] = max(averDiss.values())
        b = min(averDiss.values())
        s += (b - a)/1.0/max(a,b)

    s /= len(data)
    print 's =',s

    for c in cl.keys():
        print "Cluster ",c,", centroid ",centroids[c][0]
        count = {}
        for h in cl[c]:
            if h[0] not in count.keys():
                count[h[0]] = 1
            else:
                count[h[0]] += 1
        print "             counts: ", count

def doOneK(trainMatrix, testMatrix, k):
    counter = 0
    predictions = []
    for sample in testMatrix:
        nearestNeighbours = getNearestNeighbours( sample, trainMatrix, k)
        predictions.append( majorityVote( nearestNeighbours))
        counter = counter + 1
        if counter % 100 == 0:
            print ".",
            sys.stdout.flush()

    correct = 0
    for i in range( 0, len(predictions)):
        if predictions[ i] == testMatrix[ i][0]:
            correct = correct + 1

    accuracy = correct / (len(testMatrix) * 1.0)
    print "Test accuracy with %i-NN classifier: %f" %(k, accuracy)
    return accuracy

if __name__ == "__main__":
    main()
