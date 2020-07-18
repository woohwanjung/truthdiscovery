'''
Created on 2016. 12. 6.

@author: JWH
'''
import heapq
import numpy as np
from td_algorihm import tdalgo, normalize1d

DEBUG = False

class TDH_meta(tdalgo):
    HIERARCHY_TYPE_FULL = 0
    HIERARCHY_TYPE_CLAIMS = 1
    HIERARCHY_TYPE_CONST = 2
    HIERARCHY_TYPE_HALF = 3
    
    
    
    def __init__(self, claims, claimsByObj, names, h, popSrc = False, popWrk = True, minMaxIter = (3,25), hierarchyType = None, 
                 ubPruning = True, opt_tmp = -1, skip_single_cand = True,
                  quality_priors = 5.0, default_gamma = 1.1, regularization = True, independent_assignment = False):
        super(TDH_meta,self).__init__(claims, claimsByObj, names, h, minMaxIter = minMaxIter)
        
        if hierarchyType is None:
            hierarchyType = TDH_meta.HIERARCHY_TYPE_CLAIMS
             
        
        #TEST options
        
        self.default_gamma = default_gamma
        self.opt_tmp = opt_tmp
        self.regularization = regularization
        self.skip_single_cand_setting = skip_single_cand
        self.skip_single_cand = skip_single_cand
        
        
            
        self.popSrc = popSrc
        self.popWrk = popWrk
        
        self.states_src = 3
        self.states_wrk = 3

            
            
        self.phis = [[0.0]*self.states_src for s in range(len(self.claimsBySrc))] # 0: Correct, 1: GenCorrect, 2: Incorrect
        self.psiw = [[0.0]*self.states_wrk for w in range(self.numWorkers)]                                                  # 0: KnowCorrectly, 1: KnowGenCorrectly, 2: Incorrect, 3: Popularity
        
        self.quality_priors = quality_priors
        self.initPriors()
        
        self.value_counts = {}
        for claim in claims:
            if claim[1][1] in self.value_counts:
                self.value_counts[claim[1][1]] += 1
            else:
                self.value_counts[claim[1][1]] = 1
        
        
        
       
        self.hierarchyType = hierarchyType
        self.ubPruning  = ubPruning
        self.indepAssignment = independent_assignment
        #self._truths = [None]*len(names)
        
        self.TH_CONVERGENCE = 1E-5 * self.numObjects
        self.round = 0

    def getAlgorithmName(self):
        return "TDH_meta"
    

    
    def initPriors(self):
        self.alpha = np.full((self.numSources,self.states_src),self.quality_priors)
        self.beta = np.full((self.numWorkers,self.states_wrk),self.quality_priors)
        
        
        self.alpha[:,0] += (self.quality_priors-1.0)
        self.alpha[:,1] += (self.quality_priors-1.0)
        #self.beta[:,0] += (self.quality_priors-1.0)
        #self.beta[:,1] += (self.quality_priors-1.0)
        
        
   

    def run(self, verbose = True):
        self.begin()
        
        self.round+= 1
        
        if self.first:
            self.first = False
            self.initializeParameters()
            
        
        
        if not self.first:
            self.skip_single_cand = self.skip_single_cand_setting
        

        
        self.initilaizeEstepWorkerParameters()
        
        
        for iter in range(self.minMaxIter[1]):
            self.EstepMu()
            self.EstepPhi()
            self.EstepPsi()
            
            difference = self.MstepMu()
            self.MstepPhi()
            self.MstepPsi()
                
            #if DEBUG:
            if verbose:
                print("Iter: %d\tDiff:  %f\tTH: %f"%(iter,difference,self.TH_CONVERGENCE))
            if difference < self.TH_CONVERGENCE and iter > self.minMaxIter[0]:
                break
            
        self.setTruths()
        time_run = self.end()
        return time_run 
    
    
    def regiWorkers(self, numNewWorkers):
        super(TDH_meta,self).regiWorkers(numNewWorkers)
        self.initPriors()
        
        maxs = -1
        maxconf = 0.0
        for s in range(self.numSources):
            if self.phis[s][0] >maxconf:
                maxconf = self.phis[s][0]
                maxs = s
        
        for w in range(numNewWorkers):
            pw = [0.0]*self.states_wrk
            
            pw[0] = np.random.uniform(0.7,0.8)
            pw[1] = np.random.uniform(0.2,0.3)
            pw[2] = np.random.uniform(0.1,0.2)
            if self.states_wrk>3:
                print("num state err")
                
            normalize1d(pw)
            
            self.psiw.append(pw)
  
    def initGamma(self):
        self.gamma = []
        
        for e in range(self.numObjects):
            matchings_e = self.matching_nodes[e]
            numValues = len(self.candidates[e])
            gamma_e = np.full(numValues,self.default_gamma)
            for v in range(numValues):
                node_v = matchings_e[v]
                #gamma_e[v] += (0.5 * node_v.depth/self.h.max_depth)
            
            
            self.gamma.append(gamma_e)
            
        #self.gamma = {}
        #self.value_counts.items()
        #for k,v in self.value_counts.iteritems():
            
        #    self.gamma[k] = 2.0
            
                
                
    def initializeParameters(self):        
        #self.buildCandidatesI()
        self.initMatchingNodes()
        
        self.initGamma()
        
        self.initializeConfidences()
        self.initializeMu()
        self.initializeEstepParameters()
        self.initializePopularity()
    
    
    def initMatchingNodes(self):
        matching_nodes  = []
        count_parents = []
        count_values = []
        for e in range(self.numObjects):
            numValues = len(self.candidates[e])
            
            if self.h.preprocessed:                
                matchings_e = self.h.findingMatchingNodesPreprocessed(e, self.candidates[e])
            else:
                matchings_e = self.h.findingMatchingNodes(self.candidates[e])
            matching_nodes.append(matchings_e)
            
                
            cp_e = [0 for _ in range(numValues)]
            
            for v in range(numValues):
                node_v = matchings_e[v]
                if self.hierarchyType == TDH_meta.HIERARCHY_TYPE_CLAIMS:
                    for vp in range(numValues):
                        if v == vp:
                            continue
                        if matchings_e[vp].isAncestorof(node_v):
                            cp_e[v] += 1
                else:
                    cp_e[v] = node_v.depth
            
            if self.hierarchyType == TDH_meta.HIERARCHY_TYPE_FULL:
                count_values.append(self.h.size)
            elif self.hierarchyType == TDH_meta.HIERARCHY_TYPE_CLAIMS:
                count_values.append(numValues)
            elif self.hierarchyType == TDH_meta.HIERARCHY_TYPE_HALF:
                count_values.append(sum(cp_e)+numValues)
            elif self.hierarchyType == TDH_meta.HIERARCHY_TYPE_CONST:
                count_values.append(10)
            
            count_parents.append(cp_e)
            
        self.matching_nodes = matching_nodes
        self.count_parents = count_parents
        self.count_values = count_values

    def initializeConfidences(self):
        for s in range(self.numSources):
            self.phis[s][0] = 0.5
            self.phis[s][1] = 0.3
            self.phis[s][2] = 0.2
            #'''
            self.phis[s][0] = np.random.uniform(0.6,0.7)
            self.phis[s][1] = np.random.uniform(0.2,0.3)
            self.phis[s][2] = np.random.uniform(0.1,0.2)
            #'''
            
            normalize1d(self.phis[s])
        
        #print(self.phis)
        
        
        for w in range(self.numWorkers):
            self.psiw[w][0] = np.random.uniform(0.65,0.75)
            self.psiw[w][1] = np.random.uniform(0.15,0.25)
            self.psiw[w][2] = np.random.uniform(0.15,0.25)

            normalize1d(self.psiw[w])
            
    def initializePopularity(self):
        self.pop_weight1 = []
        self.pop_weight2 = []
        for e in range(self.numObjects):
            numClaims = float(len(self.claimsByObjI[e]) + len(self.answersByObjI[e]))
            numValues = len(self.candidates[e])
            pw1 = np.zeros((numValues,numValues))
            pw2 = np.zeros((numValues,numValues))
            
            
            
            
            for v in range(numValues):
                node_v = self.matching_nodes[e][v]
                case1 = False
                case2 = False
                for i, (s,vs) in enumerate(self.claimsByObjI[e]):
                    if vs == v:
                        continue
                    node_vs = self.matching_nodes[e][vs]
                    
                    if node_vs.isAncestorof(node_v):
                        pw1[v][vs] += 1.0
                        case1 = True
                    else:
                        pw2[v][vs] += 1.0
                        case2 = True
                
                if case1:
                    normalize1d(pw1[v])
                if case2:
                    normalize1d(pw2[v])
                
                
            self.pop_weight1.append(pw1)
            self.pop_weight2.append(pw2)
            
            
        popularity = []
        for e in range(self.numObjects):
            numClaims = float(len(self.claimsByObjI[e]) + len(self.answersByObjI[e]))
            pg = np.zeros(len(self.candidates[e]))
            for i, cand in enumerate(self.candidates[e]):
                pg[i] = cand[1]/numClaims
                
            popularity.append(pg)
        self.popularity = popularity

    def initializeMu(self):
        self.mu = []
        for e in range(self.numObjects):
            mu_e = [(self.gamma[e][v]-1.0) for v in range(len(self.candidates[e]))]
            #mu_e = [log(float(self.value_counts[candName]))+2 for candName,_ in self.candidates[e]]
            normalize1d(mu_e)
            self.mu.append(mu_e)
        self.initializeMuNumDen()    
    
    def initializeEstepParameters(self):  
        self.Pvstar_v = [np.zeros((len(self.claimsByObjI[e]),len(self.candidates[e]))) for e in range(self.numObjects)]
        self.Pb = [ np.zeros((len(self.claimsByObjI[e]),self.states_src)) for e in range(self.numObjects)]
        
    def initilaizeEstepWorkerParameters(self):
        self.Pvstar_v_w = [np.zeros((len(self.answersByObjI[e]),len(self.candidates[e]))) for e in range(self.numObjects)]
        self.PbW = [np.zeros((len(self.answersByObjI[e]),self.states_wrk)) for e in range(self.numObjects)]
        
    def setTruths(self):
        self.truths = []
        for e in range(self.numObjects):
            max_v = -1
            max_mu = -0.1
            
            if len(self.candidates[e])== 0:
                print(e,self.candidates[e],self.claimsByObj[e])
            
            for v in range(len(self.candidates[e])):
                if self.mu[e][v] >max_mu:
                    max_mu = self.mu[e][v]
                    max_v = v
            
            self.truths.append(self.candidates[e][max_v][0])
    
    def initializeMuNumDen(self):
        self.muNum = [np.zeros(len(self.candidates[e])) for e in range(self.numObjects)] 
        self.muDen = [0 for e in range(self.numObjects)] 


    def _assignTasksTDH(self, taskheap, e, eai, numQuestions):    
        if len(taskheap) < numQuestions:
            heapq.heappush(taskheap,(eai,e))
            return None
        
        if taskheap[0][0] < eai:
            lastmin = heapq.heappop(taskheap)
            heapq.heappush(taskheap,(eai,e))
            return lastmin[1]
        
        return e

    def assignTasksMaxAccuracyIncrease(self, workers, numQuestions, exp_id, cs_round):
        workerConfs = self.workerConfs(workers)
        workerConfs.sort(key=lambda v: v[1], reverse=True)
        
        answeredWorkers = [[w for w,_ in self.answersByObjI[e]] for e in range(self.numObjects)]
        
        
        upperBounds = []
        for e in range(self.numObjects):
            if len(self.candidates[e]) ==1:
                continue
            numValues = len(self.candidates[e])
            current_max_mu = max(self.mu[e])
            #De = len(self.claimsByObjI[e])+len(self.answersByObjI[e])+(numValues*1.0)
            De = self.muDen[e]
            ubEAI = (1.0-current_max_mu)/(De+1.0)
            upperBounds.append((-ubEAI, e, De, current_max_mu))#-ubEAI to make MaxHeap
            
        heapq.heapify(upperBounds)
        
        EAIheaps = [[] for _ in workers]
        tot_ub = len(upperBounds)
        while len(upperBounds)>0:
            ubEAI, e, De, current_max_mu = heapq.heappop(upperBounds)
            ubEAI = -ubEAI
            if len(EAIheaps[workerConfs[-1][0]]) >= numQuestions:
                finish_assignment = True
                for wi in range(self.numWorkers-1,-1,-1):
                    if EAIheaps[workerConfs[wi][0]][0][0] < ubEAI:
                        finish_assignment = False
                        break
                
                if finish_assignment:
                    print("%d/%d pruning ratio:%f  ub:%f"%(tot_ub-len(upperBounds),tot_ub,float(tot_ub-len(upperBounds))/tot_ub,ubEAI))
                    break
            
            
            
            for wi in range(len(workers)):
                w = workerConfs[wi][0]
                if w in answeredWorkers[e]:
                    continue
                if len(EAIheaps[w])<numQuestions or EAIheaps[w][0][0] <ubEAI:
                    EAI = self.computeExpectedAccuracyIncrease(w, e, current_max_mu)
                else:
                    EAI = ubEAI
                #if EAI >ubEAI:
                #    print("e: %d, w: %d, EAI: %f, UB_EAI: %f"%(e,w,EAI,ubEAI))
                e = self._assignTasksTDH(EAIheaps[w], e, EAI, numQuestions)
                
                if e is None:
                    break
            
            #increase = max_mu - current_max_mu
            #conf_list.append((e,increase)
        expected_total_accuracy_increase = 0.0
        for w in range(len(workers)):
            tasks = [(e, self.names[e], self.claimsByObj[e]) for eai, e  in EAIheaps[w]]
            if len(tasks) != numQuestions:
                print("Count-Pruning")
            workers[w].assignTask(tasks, exp_id, cs_round)  
            for eai, e  in EAIheaps[w]:  
                expected_total_accuracy_increase += eai
        
        expected_total_accuracy_increase/= self.numObjects
        return expected_total_accuracy_increase
    
    
    def Estep(self):
        self.EstepMu()
        self.EstepPhi()
        self.EstepPsi()
    
    def _EstepMu(self, quality, claimsByObjI, EstepParam, pop):
        for e in range(self.numObjects):   
            numValues = len(self.candidates[e])
            if self.skip_single_cand and numValues ==1:
                continue
            for i, (w,vw) in enumerate(claimsByObjI[e]):
                node_vs = self.matching_nodes[e][vw]
                sum_p = 0.0
                
                if pop:
                    for v in range(numValues):
                        if vw == v:
                            ptmp = quality[w][0]
                        
                        else:
                            node_v = self.matching_nodes[e][v]
                            if node_vs.isAncestorof(node_v):
                                ptmp = quality[w][1]*self.pop_weight1[e][v][vw]
                            else:
                                ptmp = quality[w][2]*self.pop_weight2[e][v][vw]
    
                        
                        ptmp *= self.mu[e][v]
                        EstepParam[e][i][v] = ptmp
                        sum_p += ptmp
                else:
                    for v in range(numValues):
                        if vw == v:
                            ptmp = quality[w][0]
                        
                        else:
                            node_v = self.matching_nodes[e][v]
                            if node_vs.isAncestorof(node_v):
                                ptmp = quality[w][1]/(self.count_parents[e][v])
                            else:
                                ptmp = quality[w][2]/(self.count_values[e] - self.count_parents[e][v] - 1)
                            
    
                        
                        ptmp *= self.mu[e][v]
                        EstepParam[e][i][v] = ptmp
                        sum_p += ptmp
                    
                for v in range(numValues):    
                    EstepParam[e][i][v]/= sum_p
                    

    def EstepMu(self):
        self._EstepMu(self.phis, self.claimsByObjI, self.Pvstar_v, self.popSrc)
        self._EstepMu(self.psiw, self.answersByObjI, self.Pvstar_v_w, self.popWrk)
        
    
    def EstepQuality(self, claimsByObjI, quality, estepParam, pop = False):
        numStates = 3
            
        for e in range(self.numObjects):       
            numValues = len(self.candidates[e])
            if self.skip_single_cand and numValues ==1:
                continue
            for i, (s,vs) in enumerate(claimsByObjI[e]):
                node_vs = self.matching_nodes[e][vs]

                ptmp = [0.0]*numStates
                                
                for v in range(numValues):
                    if vs == v:
                        ptmp[0] = self.mu[e][v]
                    elif pop:
                        node_v = self.matching_nodes[e][v]
                        if node_vs.isAncestorof(node_v):
                            ptmp[1] += (self.mu[e][v]*self.pop_weight1[e][v][vs])
                        else:
                            ptmp[2] += (self.mu[e][v]*self.pop_weight2[e][v][vs])
                    else:
                        node_v = self.matching_nodes[e][v]
                        if node_vs.isAncestorof(node_v):
                            ptmp[1] += (self.mu[e][v]/self.count_parents[e][v])
                        else:
                            ptmp[2] += (self.mu[e][v]/(self.count_values[e] - self.count_parents[e][v] - 1))
                            
                    
                        
                for k in range(numStates):
                    ptmp[k] *= quality[s][k]
                     
                ptmp_sum = sum(ptmp)
                for k in range(numStates):
                    estepParam[e][i][k] = ptmp[k]/ptmp_sum
                    
        

            
    def EstepPhi(self):
        self.EstepQuality(self.claimsByObjI, self.phis, self.Pb, self.popSrc)       
                    
    def EstepPsi(self):
        self.EstepQuality(self.answersByObjI, self.psiw, self.PbW, self.popWrk)
        
    
    
    def MstepMu(self):
        delta_mu_sum = 0.0
        for e in range(self.numObjects):
            numValues  = len(self.candidates[e])
            if self.skip_single_cand and self.single_cand[e]:
                continue
            
            self.muDen[e] = len(self.claimsByObjI[e])+len(self.answersByObjI[e])
            
            #g1_list = []
            muNum = self.muNum[e]
            #for v, (candName, candCount) in enumerate(self.candidates[e]):
            
            if self.regularization:
                for v in range(numValues):
                    g1 = self.gamma[e][v]-1.0
                    muNum[v] = g1
                    self.muDen[e] += (g1)
            else:
                muNum.fill(0.0)
                
            
            for v in range(numValues):
                mu_prev = self.mu[e][v]
                
                for i, (s,vs) in enumerate(self.claimsByObjI[e]):
                    muNum[v] += self.Pvstar_v[e][i][v]
                    
                for i, (w,vw) in enumerate(self.answersByObjI[e]):
                    muNum[v] += self.Pvstar_v_w[e][i][v]
                
                self.mu[e][v] =  muNum[v]/ self.muDen[e]
                
                delta_mu_sum += abs(mu_prev-self.mu[e][v])
        return delta_mu_sum
    
    def MstepQuality(self,quality, prior, claimsByObjI, estepPrams, numSources, pop):
        numStates = 3
        
        if self.regularization:
            for s in range(numSources):
                for b in range(numStates):
                    quality[s][b] = prior[s][b] - 1.0
            
        for e in range(self.numObjects):
            numValues = len(self.candidates[e])
            if self.skip_single_cand and self.single_cand[e]:
                continue
            
            for i, (s,vs) in enumerate(claimsByObjI[e]):
                for b in range(numStates):
                    quality[s][b] += estepPrams[e][i][b]
                
        
        
        for s in range(numSources):
            
            normalize1d(quality[s])
    
    def MstepPhi(self):
        self.MstepQuality(self.phis, self.alpha, self.claimsByObjI, self.Pb, self.numSources, self.popSrc)
    
    def MstepPsi(self):
        self.MstepQuality(self.psiw, self.beta,self.answersByObjI, self.PbW, self.numWorkers, self.popWrk)

    def Mstep(self):
        self.MstepMu()
        self.MstepPhi()
        self.MstepPsi()
                    
    
    def getConfidences(self,e):
        return self.mu[e]
    
    def getVotes(self,e):
        conf_list = [ len(self.mu[e]) * v for v in self.mu[e]]
        return conf_list
    
    def computeExpectedAccuracyIncrease(self, w, e, current_maxmu):
        pv      = [0.0 for _ in range(len(self.candidates[e]))]
        pv_vp   = [[0.0 for _ in range(len(self.candidates[e]))] for _ in range(len(self.candidates[e]))]
        numValues = len(self.candidates[e])
        #compute Pv
        for vp in range(numValues):
            for v in range(numValues):
                if v == vp:
                    tmp = self.psiw[w][0] * self.mu[e][v]
                    if not self.descendant_ancestor_relationship[e]:
                        tmp += self.psiw[w][1] * self.mu[e][v]
                elif self.popWrk:
                    if self.matching_nodes[e][vp].isAncestorof(self.matching_nodes[e][v]):
                        tmp = self.psiw[w][1] *self.mu[e][v] *self.pop_weight1[e][v][vp]
                    else:
                        tmp = self.psiw[w][2] *self.mu[e][v] *self.pop_weight2[e][v][vp]
                else:
                    if self.matching_nodes[e][vp].isAncestorof(self.matching_nodes[e][v]):
                        tmp = self.psiw[w][1] / self.count_parents[e][v]*self.mu[e][v]
                    else:
                        tmp = self.psiw[w][2] / (self.count_values[e] -self.count_parents[e][v]-1.0)*self.mu[e][v]
                pv[vp] += tmp
                pv_vp[v][vp] = tmp
                
        sumpv = sum(pv) ######Detailed Implementation sum(pv) is not equal to 1 if not every case 1,2,3 is possible
        
        psum = 0.0
        for vp in range(numValues):
            maxp = 0.0
            for v in range(numValues):
                mu_cond = (self.muNum[e][v] + pv_vp[v][vp]/pv[vp])/(self.muDen[e]+1.0)
                
                #tmp = pv[vp]/sumpv*self.muNum[e][v] + pv_vp[v][vp]/sumpv
                if mu_cond > maxp:
                    maxp = mu_cond
            psum += (maxp*pv[vp]/sumpv)
        
        #psum/= (self.muDen[e]+1.0)
        increase = psum-current_maxmu
        if DEBUG:
            print("Mu: %f/%f   increase: %f"%(psum,current_maxmu,increase))
        return increase
    
    
        
    def workerConfs(self, workers):
        return [(worker.wid,self.psiw[worker.wid][0]) for worker in workers]
        
    
    def get_current_distribution_matrix(self):
        return self.mu
    
    def check_single_cand(self):
        self.single_cand = [True]*self.numObjects
        for e in range(self.numObjects):
            numValues = len(self.candidates[e])
            if numValues == 1:
                continue
            
            if self.numericdata:
                fval = self.candidates[e][0][0]
                for (cand_v, cand_sig),_ in self.candidates[e]:
                    if fval != cand_v:
                        self.single_cand[e] = False
                        break
            else:
                self.single_cand[e] = False



class TDH(TDH_meta):
    def __init__(self, claims, claimsByObj, names, h, minMaxIter = (3,25), hierarchyType = None, 
                 ubPruning = True, skip_single_cand = False,
                 popWrk = False, popSrc = False,quality_priors = 2.0, default_gamma = 2.0, regularization = True, numericdata = False):
        
        super(TDH,self).__init__(claims, claimsByObj, names, h, popSrc = popSrc, popWrk = popWrk, minMaxIter = minMaxIter,
                hierarchyType=hierarchyType, ubPruning= ubPruning, opt_tmp = 0, skip_single_cand = skip_single_cand,
                quality_priors= quality_priors, default_gamma =default_gamma, regularization=regularization)
    
        self.numericdata = numericdata
    
    def initializeParameters(self):
        #self.gamma[k] += v/val_count_sum
        
        #self.buildCandidatesI()
        self.initMatchingNodes()
        self.initGamma()
        self.initializeConfidences()
        self.initializeMu()
        self.initializeEstepParameters()
        self.initializePopularity()
        self.check_desc_acs()
        self.check_single_cand()
    

    
    
    def check_desc_acs(self):
        self.descendant_ancestor_relationship  = [False]*self.numObjects
        for e in range(self.numObjects):
            numValues = len(self.candidates[e])
            for i in range(numValues):
                for j in range(i+1, numValues):
                    if self.matching_nodes[e][i].isAncestorof(self.matching_nodes[e][j]):
                        self.descendant_ancestor_relationship[e] = True
                        break
                    if self.matching_nodes[e][j].isAncestorof(self.matching_nodes[e][i]):
                        self.descendant_ancestor_relationship[e] = True
                        break
                
                if self.descendant_ancestor_relationship[e]:
                    break
            
        
    
    
    def EstepMu(self):
        self._EstepMu(self.phis, self.claimsByObjI, self.Pvstar_v, self.popSrc)
        self._EstepMu(self.psiw, self.answersByObjI, self.Pvstar_v_w, self.popWrk)
        
    def EstepQuality(self, claimsByObjI, quality, estepParam, pop = False):
        for e in range(self.numObjects):       
            numValues = len(self.candidates[e])
            if self.skip_single_cand and self.single_cand[e]:
                continue
            for i, (s,vs) in enumerate(claimsByObjI[e]):
                ptmp = estepParam[e][i]
                ptmp.fill(0.0)
                
                if self.descendant_ancestor_relationship[e]:
                    node_vs = self.matching_nodes[e][vs]
                    for v in range(numValues):
                        if vs == v:
                            ptmp[0] = self.mu[e][v]
                        elif pop:
                            node_v = self.matching_nodes[e][v]
                            if node_vs.isAncestorof(node_v):
                                ptmp[1] += (self.mu[e][v]*self.pop_weight1[e][v][vs])
                            else:
                                ptmp[2] += (self.mu[e][v]*self.pop_weight2[e][v][vs])
                        else:
                            node_v = self.matching_nodes[e][v]
                            if node_vs.isAncestorof(node_v):
                                ptmp[1] += (self.mu[e][v]/self.count_parents[e][v])
                            else:
                                ptmp[2] += (self.mu[e][v]/(self.count_values[e] - self.count_parents[e][v] - 1))
                        
                else:
                    for v in range(numValues):
                        if vs == v:
                            ptmp[0] = self.mu[e][v]
                            ptmp[1] = self.mu[e][v]
                        elif pop:
                            ptmp[2] += (self.mu[e][v]*self.pop_weight2[e][v][vs])
                        else:
                            ptmp[2] += (self.mu[e][v]/(self.count_values[e]- self.count_parents[e][v] - 1))
                            
                        
                for k in range(3):
                    ptmp[k] *= quality[s][k]
                     
                normalize1d(ptmp)
    
    def _EstepMu(self, quality, claimsByObjI, EstepParam, pop):
        for e in range(self.numObjects):   
            numValues = len(self.candidates[e])
            if self.skip_single_cand and self.single_cand[e]:
                continue
            for i, (w,vw) in enumerate(claimsByObjI[e]):
                sum_p = 0.0
                
                if self.descendant_ancestor_relationship[e]:
                    node_vs = self.matching_nodes[e][vw]
                    for v in range(numValues):
                        if vw == v:
                            ptmp = quality[w][0]
                        elif pop:
                            node_v = self.matching_nodes[e][v]
                            if node_vs.isAncestorof(node_v):
                                ptmp = quality[w][1]*self.pop_weight1[e][v][vw]
                            else:
                                ptmp = quality[w][2]*self.pop_weight2[e][v][vw]
                        else:
                            node_v = self.matching_nodes[e][v]
                            if node_vs.isAncestorof(node_v):
                                ptmp = quality[w][1]/(self.count_parents[e][v])
                            else:
                                ptmp = quality[w][2]/(self.count_values[e] - self.count_parents[e][v] - 1)
                            
                        
                        ptmp *= self.mu[e][v]
                        EstepParam[e][i][v] = ptmp
                        sum_p += ptmp
                else:
                    for v in range(numValues):
                        if vw == v:
                            ptmp = quality[w][0]+quality[w][1]
                        elif pop:
                            ptmp = quality[w][2]*self.pop_weight2[e][v][vw]
                        else:
                            ptmp = quality[w][2]/(self.count_values[e]- self.count_parents[e][v] - 1)
                        
                        ptmp *= self.mu[e][v]
                        EstepParam[e][i][v] = ptmp
                        sum_p += ptmp
                if sum_p == 0:
                    EstepParam[e][i] = 1.0/numValues
                else:
                    EstepParam[e][i]/= sum_p
                 
             
    def get_estimated_distribution_matrix(self, w): 
        answered = set([answer[0] for answer in self.answersByWorkerI[w]])
        
        estimated_distribution_list = []
        for e in range(self.numObjects):
            if e in answered:
                continue
            
            
            estmated_distribution = np.zeros(len(self.mu[e]))
            numCand = len(estmated_distribution)
            if numCand == 1:
                continue
            
            #step 1            
            p_at_rev = np.zeros((numCand,numCand))
            p_a = np.zeros((numCand))
            for t in range(numCand):
                node_t = self.matching_nodes[e][t]
                if self.descendant_ancestor_relationship[e]:
                    for a in range(numCand):
                        if a == t:
                            p_tmp = self.psiw[w][0]
                        elif self.popWrk:
                            node_a = self.matching_nodes[e][a]
                            if node_a.isAncestorof(node_t):
                                p_tmp = self.psiw[w][1]*self.pop_weight1[e][t][a]
                            else:
                                p_tmp = self.psiw[w][2]*self.pop_weight2[e][t][a]
                        else:
                            node_a = self.matching_nodes[e][a]
                            if node_a.isAncestorof(node_t):
                                p_tmp = self.psiw[w][1]/(self.count_parents[e][t])
                            else:
                                p_tmp = self.psiw[w][2]/(self.count_values[e] - self.count_parents[e][t] - 1)
                        p_at_rev[t][a] = p_tmp
                        
                else:
                    for a in range(numCand):
                        if a == t:
                            p_tmp = self.psiw[w][0] +self.psiw[w][1]
                        elif self.popWrk:
                            p_tmp = self.psiw[w][2]*self.pop_weight2[e][t][a]
                        else:
                            p_tmp = self.psiw[w][2]/(self.count_values[e] - self.count_parents[e][t] - 1)
                        p_at_rev[t][a] = p_tmp
                
                normalize1d(p_at_rev[t])
                for a in range(numCand):
                    p_a[a] += p_at_rev[t][a]*self.mu[e][t]
            
            if sum(p_a)>1.01 or sum(p_a)<0.99:
                print("sum")               
            
            li = np.random.choice(range(numCand),p = p_a)
            
            #step 2
            for t in range(numCand):
                estmated_distribution[t] = self.mu[e][t] *p_at_rev[li][t]
                
            normalize1d(estmated_distribution)
            estimated_distribution_list.append((e,estmated_distribution))       
        return estimated_distribution_list







                
                
if __name__ == '__main__':
    #fullEXP()

    #testing2phase()
    #testing_reg()
    #expConfidence("birthPlaces")
    #expConfidence("whc")
    pass