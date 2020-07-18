'''
Created on 2017. 1. 26.

@author: JWH
'''
from abc import ABCMeta
import os
import random


import numpy as np



def buildSimulatedWorkers(nW, groundTruth, phi = None):
    if phi is None:
        phi = 0.8
    if type(phi) is float:
        phi = [phi]*nW
    
    workers = [SimulatedWorker(w,phi[w],groundTruth) for w in range(nW)]
    return workers



def loadSimulatedWorkersUniform(prob_min, prob_max, numWorkers, groundTruth):
    prob = np.zeros(numWorkers)
    
    interval = prob_max-prob_min
    
    for w in range(numWorkers):
        prob[w] = prob_min + interval*random.random()
        if prob[w] >prob_max or prob[w] <prob_min:
            print("ERRRR")
    
    
    
    workers = [SimulatedWorker(w,prob[w],groundTruth) for w in range(numWorkers)]
    return workers    






class Worker:
    __metaclass__=ABCMeta
    
    def __init__(self, wid):
        self.wid = wid
        self.tasks = []
        self.newtask = None
        
    
    def getAnswer(self):
        pass
    
    def answered(self):
        pass
    
    def buildCandidates(self):
        self.candidates = [None for q in range(len(self.newtask))]
        for q in range(len(self.newtask)):
            question = self.newtask[q]
            
            cands = {}
            for claim in question[2]:
                if claim[1] in cands:
                    cands[claim[1]].append(claim[0])
                else:
                    cands[claim[1]]=[claim[0]]
                    
            self.candidates[q] = list(cands.items())
            
            

    
class SimulatedWorker(Worker):
    def __init__(self, wid, phi, ground_truths, pGen = 0.0, sampleFromClaims = False):
        super(SimulatedWorker,self).__init__(wid)
        self.ground_truths = ground_truths
        self.phi = phi
        self.pGen = pGen
        self.sampleFromClaims = sampleFromClaims
        
    def assignTask(self, task, expid, csround):
        #print("Assgin a task to Simulated Worker "+str(self.id))
        self.newtask = task
        self.tasks.append(task)
        self.buildCandidates()
        
        
    def getAnswer(self, ):
        #Worker.getAnswer(self)
        answer = []
        for q in range(len(self.newtask)):
            e, name, _ = self.newtask[q]
            candidates = self.candidates[q]
            if np.random.rand() > self.phi: #Draw by random
                cand_vals = []
                for cv, sources in candidates:
                    if self.sampleFromClaims:
                        for s in sources:
                            cand_vals.append(cv)
                    else:
                        cand_vals.append(cv)
                answer.append((e,np.random.choice(cand_vals)))
                continue
            
            genAnswer = np.random.rand() <= self.pGen
            answered = False
            genCands = []
            truth = self.ground_truths[name]
            for v, sources in candidates:
                if v == truth[0]:
                    if not genAnswer:
                        answer.append((e,v))
                        answered = True
                        break
                    
                for vt in truth[1:]:
                    if v == vt:
                        genCands.append(v)
                        break
            
            if not answered:
                if len(genCands) == 0:
                    answer.append((e,truth[0]))
                else:
                    answer.append((e,np.random.choice(genCands)))

        self.newtask = None
        return answer
    
    def answered(self):
        return True
    

        
    


        
def testWorker():
    task1 = [(0, 'DiCaprio, Leonardo',[(0,'Los Angeles'), (1,'Los Angeles'), (2,'Hollywood')])]
    task2 = [(0, 'DiCaprio, Leonardo',[(0,'Los Angeles'), (2,'China'), (3,'Korea')])]
    
    sw = SimulatedWorker(1, 0.8, {'DiCaprio, Leonardo':['sdf','Hollywood','Los Angeles','California','USA']})
   
    
    
    
    print("Task 1")
    countTrue = 0
    repetition = 10000
    for i in range(repetition):
        sw.assignTask(task1)
        #print task1
        #print sw.ground_truths[task1[0][0][0:]]
        #print sw.getAnswer()
        answer  = sw.getAnswer()
        if answer[0][1] == 'Hollywood':
            countTrue +=1
    print ("count: %d\t expected: %d"%(countTrue,repetition*0.9))
    print("Task 2")
    countTrue = 0
    for i in range(repetition):
        sw.assignTask(task2)
        #print task2
        #print sw.ground_truths[task2[0][0][0:]]
        answer  = sw.getAnswer()
        if answer[0][1] == 'Los Angeles':
            countTrue +=1
            
    print ("count: %d\t expected: %d"%(countTrue,repetition*0.866666))    



def groupByEntity(res):
    e2tids = {}
    for tid, entity, eid in res:
        if entity in e2tids:
            e2tids[entity].append((tid,eid))
        else:
            e2tids[entity] = [(tid,eid)]
    return e2tids



if __name__ == '__main__':
    '''
    acc_list = [(0.5,1.0),(0.9,1.0),(0.0,0.1),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.6,0.9),(0.5,1.0)]
    for prob_min, prob_max in acc_list:
        #saveSimulatedWorkersUniform(prob_min, prob_max)
        saveSimulatedWorkersUniformDomainSensitive(prob_min, prob_max, 2)
        saveSimulatedWorkersUniformDomainSensitive(prob_min, prob_max, 3)
    '''
    #setReusable()

    #userAccuracy("birthPlaces")
    pass