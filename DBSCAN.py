import matplotlib.pyplot as plt
import numpy as np
import random
import math
from queue import Queue
from sklearn import datasets

# NSamples = 100
NSamples = 150
np.random.seed(42)
(DToken,DLabel) = datasets.make_moons(n_samples=NSamples, noise=0.05)


def DrawTestPic(D):

    for (x,y) in D:
        plt.scatter(x,y,c="tab:blue",s=4)
    plt.show()


def DBSCAN(Data,Epsilon,MinPts):

    def CalDistance(Sample1, Sample2):
        """
        Calculate Euclidean distance of Sample1 and Sample2.
        :return:Distance result
        """
        return math.sqrt(((Sample1[0] - Sample2[0]) ** 2) + (Sample1[1] - Sample2[1]) ** 2)

    def GetEpsilonNeighborhood(SamplePoint, Data, Epsilon):
        """
        The simplest way to find,We can use KD-Tree to reduce TimeComplexity.
        :param Sample: SamplePoint
        :return: Epsilon neighborhood of SamplePoint.
        """
        EpsilonNeighList = list()
        for K in Data:
            if tuple(K) == tuple(SamplePoint):
                continue
            else:
                DistKSamplePoint = CalDistance(K, SamplePoint)
                if DistKSamplePoint < Epsilon:
                    EpsilonNeighList.append(tuple(K))
                    
        return EpsilonNeighList

    def DiscrimCoreObject(Data,Epsilon,MinPts):
        """
        Get list of core object.
        :return: CoreObjectList.
        """
        CoreObjectList = list()
        for Sample in Data:
            EpsilonNeighList = GetEpsilonNeighborhood(Sample,Data,Epsilon)
            if len(EpsilonNeighList) >= MinPts:
                CoreObjectList.append(tuple(Sample))

        return CoreObjectList

    def ExecutiveProcess(Data,CoreObjectList):

        def CalRelDifferenceSet(List1,List2):
            """
            Calculate Relative Difference Set of List1 and List2
            :return:
            """
            return list(set(List1).difference(set(List2)))

        def CalIntersection(List1,List2):
            """
            Calculate Intersection Set of List1 and List2.
            :return:
            """
            return list(set(List1).intersection(set(List2)))

        ClusterNum = 0
        ClusterList = list()

        Gamma = [tuple(i) for i in Data.tolist()]
        while CoreObjectList:
            GammaOld = [tuple(i) for i in Gamma]
            RandomIndex = random.randint(0,len(CoreObjectList) - 1)
            RandomCoreObject = CoreObjectList[RandomIndex]
            Q = Queue()
            Q.put(RandomCoreObject)
            Gamma = CalRelDifferenceSet(Gamma,[RandomCoreObject])

            while not Q.empty():
                FirstElem = Q.get()
                EpsilonNeighborL = GetEpsilonNeighborhood(FirstElem,Data,Epsilon)
                if len(EpsilonNeighborL) >= MinPts:
                    Delta = CalIntersection(EpsilonNeighborL,Gamma)
                    for Elem in Delta:
                        Q.put(Elem)
                    Gamma = CalRelDifferenceSet(Gamma,Delta)

            NewCluster = CalRelDifferenceSet(GammaOld,Gamma)
            ClusterList.append(NewCluster)
            CoreObjectList = CalRelDifferenceSet(CoreObjectList,NewCluster)
            ClusterNum += 1

        return ClusterNum,ClusterList

    def PaintCluster(ClusterList):
        ColorList = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                     "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
        for idx,Cluster in enumerate(ClusterList):
            for (x1,x2) in Cluster:
                plt.scatter(x1,x2,s=4,c=ColorList[idx])
        plt.show()

    ClusterNum,ClusterList = ExecutiveProcess(Data, DiscrimCoreObject(Data, Epsilon, MinPts))
    print(ClusterNum)
    return PaintCluster(ClusterList)

