'''
Created on 2016. 11. 10.

@author: JWH
'''
import datetime
from math import log
from operator import itemgetter

from dataloader import recordsBySource

DEBUG = False
#DEBUG = True




def cliping(array, max_v = 10.0, min_v = -10.0):
    for i in range(len(array)):
        if array[i ] > max_v:
            array[i] = max_v
        elif array[i] <min_v:
            array[i] = min_v

def normalize1d(vec):
    sum_p = 0.0
    for v in vec:
        sum_p += v
    if sum_p==0.0:
        for i in range(len(vec)):
            vec[i]= 1.0/len(vec)
        
    else:      
        for i in range(len(vec)):
            vec[i]/=sum_p
        


def computeEntropy(vec):
    entropy = 0.0
    for p in vec:
        if p >.0:
            entropy -= p*log(p,2.0)
    return entropy

    
class IncompatibleAlgorithmException(Exception):
    pass

class tdalgo(object):
    '''
    classdocs
    '''
    TASK_ASSIGNMENT_MAX_ENTROPY = 0
    TASK_ASSIGNMENT_MAX_ACCURACY_INCREASE = 1
    TASK_ASSIGNMENT_MAX_BENEFIT = 2
    TASK_ASSIGNMENT_QASCA = 3
    
    
    TASK_ASSIGNMENT_NAME = ['MAX_ENTROPY' ,'MAX_ACC_INCREASE','MAX_BENEFIT','QASCA']
    
    def __init__(self,claims, claimsByObj, names, h, minMaxIter = (3,25)):
        '''
        Constructor
        '''
        self.minMaxIter = minMaxIter
        self.claims = claims
        self.h = h
        self.numObjects = len(names)
        self.names = names
        
        self.claimsByObj = claimsByObj
        self.claimsBySrc = recordsBySource(claims)
        
        
        self.numSources = len(self.claimsBySrc)

        self.answersByWorker = []
        self.answersByObj = [[] for e in range(self.numObjects)]
        self.answersByWorkerI = []
        self.answersByObjI = [[] for e in range(self.numObjects)]
        
        self.numWorkers = len(self.answersByWorker)
        self.buildCandidatesI()
        
        self.first = True
    
    
    def getAlgorithmName(self):
        raise Exception("getAlgorithm name is not implemented "+str(self.__class__))
    
    def buildCandidates(self):
        candidates = [{} for e in range(self.numObjects)]
        for e in range(self.numObjects):
            for claim in self.claimsByObj[e]:
                if claim[1] in candidates[e]:
                    candidates[e][claim[1]] += 1
                else:
                    candidates[e][claim[1]] = 1
        return candidates

    def buildCandidatesI(self):
        candidates = self.buildCandidates()
        
        self.candidates = [sorted(cand_e.items(),key = lambda rec:rec[1],reverse=True) for cand_e in candidates]
        
        for e in range(self.numObjects):
            for c in range(len(self.candidates[e])):
                self.candidates[e][c] = list(self.candidates[e][c])
        
        
        inverseCand = [{} for e in range(self.numObjects)]
        for e in range(self.numObjects):
            for i, (candname, cand_count) in enumerate(self.candidates[e]):
                inverseCand[e][candname] = i
        
        self.claimsByObjI = [[] for e in range(self.numObjects)]
        self.claimsBySrcI = [[] for s in range(self.numSources)]
        for e in range(self.numObjects):
            for claim in self.claimsByObj[e]:
                s = claim[0]
                val = claim[1]
                candId = inverseCand[e][val]
                self.claimsByObjI[e].append((s,candId))
                self.claimsBySrcI[s].append((e,candId))
                
    
    def _regiAnswers(self, wid, answers):
        for answer in answers:
            e, v  = answer
            matched = False
            for c in range(len(self.candidates[e])):
                if self.candidates[e][c][0] == v:
                    self.candidates[e][c][1] +=1
                    self.answersByWorkerI[wid].append((e,c))
                    self.answersByObjI[e].append((wid,c))
                    self.answersByObj[e].append((wid,v))
                    self.answersByWorker[wid].append((e,v))
                    matched = True
                    break
            if not matched:
                print("regiAnswer Err")
            
    def regiAnswers(self, workers):
        for worker in workers:
            answers = worker.getAnswer()
            wid = worker.wid
            self._regiAnswers(wid, answers)
            
    
            
        
    def regiWorkers(self,numNewWorkers):
        for _ in range(numNewWorkers):
            self.answersByWorker.append([])
            self.answersByWorkerI.append([])
        self.numWorkers += numNewWorkers
            

    def getTruths(self):
        return self.truths
    
    def begin(self):
        self.t_begin = datetime.datetime.now()
    def end(self):
        self.t_end = datetime.datetime.now()
        td = self.t_end-self.t_begin
        print(td)
        return td
        
    def assignTasks(self, workers, numQuestions, AT_TYPE, exp_id = -1, cs_round = -1, get_stat = False):
        t_begin = datetime.datetime.now()
        if AT_TYPE == tdalgo.TASK_ASSIGNMENT_MAX_ENTROPY:
            stat = self.assignTasksMaxEntropy(workers, numQuestions, exp_id, cs_round)
        elif AT_TYPE == tdalgo.TASK_ASSIGNMENT_MAX_ACCURACY_INCREASE:
            stat = self.assignTasksMaxAccuracyIncrease(workers, numQuestions, exp_id, cs_round)
        elif AT_TYPE == tdalgo.TASK_ASSIGNMENT_MAX_BENEFIT:
            stat = self.assignTasksMaxBenefit(workers, numQuestions, exp_id, cs_round)
        elif AT_TYPE ==tdalgo.TASK_ASSIGNMENT_QASCA:
            stat = self.assignTasksQASCA(workers, numQuestions, exp_id, cs_round)
        else:
            raise Exception()
        t_end = datetime.datetime.now()
        td = t_end-t_begin
        if get_stat:
            return td, stat
            
        return td
    
    def _assignTasks(self, workers, numQuestions, scoreList, experiment_id, crowdsourcing_round, descending = False):
        scoreList.sort(key=lambda v:v[1], reverse=descending)

        tot_tasks = numQuestions* len(workers)
        tasks = [[]for q in range(len(workers))]
        i = 0
        count_assign = 0
        while count_assign < tot_tasks and i < len(scoreList):
            e = scoreList[i][0]
            for w in range(len(workers)):
                if len(tasks[w]) >= numQuestions:
                    continue
                answered = False
                for  _w , _ in self.answersByObjI[e]:
                    if w == _w:
                        answered = True
                        break
                if answered:
                    continue
                    
                tasks[w].append((e, self.names[e], self.claimsByObj[e]))
                count_assign+= 1
                break
            i+=1

            
        for w in range(len(workers)):
            workers[w].assignTask(tasks[w], experiment_id, crowdsourcing_round)
        
        
    def assignTasksMaxEntropy(self, workers, numQuestions, exp_id = -1, cs_round = -1):
        ent_list =[]
        for e in range(self.numObjects):
            confidences = self.getConfidences(e)
            if len(confidences) <=1:
                continue
            
            normalize1d(confidences)
            entropy = computeEntropy(confidences)
            
            ent_list.append((e,entropy))
        
        self._assignTasks(workers, numQuestions, ent_list, descending = True , experiment_id = exp_id, crowdsourcing_round = cs_round)
        

    def assignTasksMaxAccuracyIncrease(self, workers, numQuestions, exp_id = -1, cs_round = -1):
        raise IncompatibleAlgorithmException()
    
 
    def assignTasksMaxBenefit(self, workers, numQuestions, exp_id = -1, cs_round = -1):
        raise IncompatibleAlgorithmException()
    
    
    
    def assignTasksQASCA(self, workers, numQuestions, exp_id = -1, cs_round = -1):
        assigned = set()
        
        tasks = [[]for q in range(len(workers))]
        
        cdm = self.get_current_distribution_matrix()
        
        expected_total_accuracy_increase = 0.0
        
        for w in range(len(workers)):
            cdm_w = self.get_estimated_distribution_matrix(w)
            
            candidate_list = []
            
            for e, estimated_distribution in cdm_w:
                if e in assigned:
                    continue 
                
                benefit = max(estimated_distribution) - max(cdm[e])
                
                candidate_list.append((benefit,e))
            
            candidate_list.sort(key = itemgetter(0),reverse=True) 
            
            for q in range(numQuestions):
                benefit, e = candidate_list[q]
                assigned.add(e)
                tasks[w].append((e,self.names[e], self.claimsByObj[e]))
                expected_total_accuracy_increase+= benefit
        
        for w in range(len(workers)):
            workers[w].assignTask(tasks[w], exp_id, cs_round) 
        
        expected_total_accuracy_increase/= self.numObjects
        return expected_total_accuracy_increase 
    

  

class Vote(tdalgo):
    def __init__(self,claims, claimsByObj, names, h, minMaxIter= None):
        super(Vote,self).__init__(claims, claimsByObj, names, h, minMaxIter = minMaxIter)
    
    def getAlgorithmName(self):
        return "Vote"

    def run(self, verbose = False):
        self.begin()
        if self.first:
            self.truths = [None]*self.numObjects
            self.first = False
        
        for e in range(self.numObjects):
            counts = {}
            for claim in self.claimsByObj[e]:
                if claim[1] in counts:
                    counts[claim[1]] +=1
                else:
                    counts[claim[1]] = 1
            for claim in self.answersByObj[e]:
                if claim[1] in counts:
                    counts[claim[1]] +=1
                else:
                    counts[claim[1]] = 1
                    
            maxcount = -1
            for v, count in counts.items():
                if count > maxcount:
                    self.truths[e] = v
                    maxcount = count

        time_run = self.end()
        return time_run
    
    def getVotes(self,e):
        conf_list = [float(cand[1]) for cand in self.candidates[e]]
        return conf_list 
    
    def getConfidences(self,e):
        conf_list = self.getVotes(e)
        normalize1d(conf_list)
        return conf_list 
    
    
