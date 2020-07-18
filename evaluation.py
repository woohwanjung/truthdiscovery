

EVAL_OPTION_ALL = 1
EVAL_OPTION_EXIST = 2
class EvaluationModule():
    def __init__(self, ground_truths, names, claimsByObj, h, eval_option = EVAL_OPTION_EXIST):
        self.ground_truths = ground_truths
        self.names = names
        self.claimsByObj = claimsByObj
        self.h = h
        self.init_gt_height()
        self.eval_option = eval_option
        
    def init_gt_height(self):
        self.specific_candidate_height = []
        for e in range(len(self.names)):
            name = self.names[e]
            #try:
            
            gt = self.ground_truths[name]
          
            exists = False
            for claim in self.claimsByObj[e]:
                if gt[0] == claim[1]:
                    exists = True
                    self.specific_candidate_height.append(0)
                    break
            
            if not exists:
                for d in range(1,len(gt)):
                #for bp in birthplace[1:]:
                    bp = gt[d]
                    for claim in self.claimsByObj[e]:
                        if bp == claim[1]:
                            exists = True
                            break
                    if exists:
                        self.specific_candidate_height.append(d)
                        break
        
        
    def eval(self, est_truths):
        correct_count = 0
        correct_count_gen = 0
        
        distance_t2e = 0
        

        #print(ground_truths)
        for e in range(len(self.names)):
            name = self.names[e]
            if self.eval_option == EVAL_OPTION_EXIST:
                gt = self.ground_truths[name][self.specific_candidate_height[e]:]
            elif self.eval_option == EVAL_OPTION_ALL:
                gt = self.ground_truths[name]
            
            
            et = est_truths[e]
            if et == gt[0]:
                correct_count +=1
            else:
                for i in range(1,len(gt)):
                    if et == gt[i]:
                        correct_count_gen += 1
                        break
            
                nodes_truths = self.h.getNodes(gt[0])
                nodes_est_truths = self.h.getNodes(et)
                if nodes_truths is None:
                    print("None")
                    print(gt)
                    print(et)
                    print(e)
                    continue
                min_dist = 100
                for node_t in nodes_truths:
                    for node_et in nodes_est_truths:
                        dist = self.h.computeDistance(node_t, node_et)
                        min_dist = min(dist, min_dist)
                distance_t2e+=min_dist
                
                if min_dist == 100 or min_dist <1:
                    print("Distance0")
                
                
                
                
        correct_count_gen += correct_count
        
        
        accuracy = float(correct_count)/len(self.names)
        accuracy_gen = float(correct_count_gen)/len(self.names)
        avg_distance = float(distance_t2e)/len(self.names)
        
        print("%12s: %f (%d/%d)"%("Accuracy",accuracy,correct_count,len(self.names)))
        print("%12s: %f (%d/%d)"%("AccuracyGen",accuracy_gen,correct_count_gen,len(self.names)))        
        print("%12s: %f (%d/%d)"%("AvgDistance",avg_distance,distance_t2e,len(self.names)))  
        return accuracy, accuracy_gen, avg_distance