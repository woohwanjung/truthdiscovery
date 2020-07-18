from TDH import TDH
from crowdsourcing import loadSimulatedWorkersUniform
from dataloader import loadData, deepCopyClaimsByObj
from evaluation import EvaluationModule
from td_algorihm import Vote

import argparse


DEBUG = False

#Crowdsourcing setting
num_workers = 10
num_rounds = 50
numQuestions = 5



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--data", choices = ["birthplaces","heritages"], default = "heritages", help = "Datasets")
parser.add_argument('--truth_inference', choices = ['TDH','Vote'], default = "TDH", help = "Truth inference model:  TDH|Vote")
parser.add_argument("--crowdsourcing", type = str2bool, default = False)
parser.add_argument("--task_assignment", choices = ["ME","EAI"], default = "ME", help = "Task assignment algorithm:  ME|EAI")
parser.add_argument("--verbose", type = str2bool, default = False)
parser.add_argument("--min_iter", type = int, default = 3, help = "Minimum number of iterations in a round")
parser.add_argument("--max_iter", type = int, default = 50, help = "Maximum number of iterations in a round")

if __name__ == "__main__":
    args = parser.parse_args()
    #Settings
    min_max_iter = (args.min_iter,args.max_iter)
    
    #Load data
    dataname = args.data
    print(f"Truth inference: {args.truth_inference}")
    if args.crowdsourcing:
        print(f"Task assignment: {args.task_assignment}")
    
    print(f"Data: {dataname}")
    claims, claimsByObj, names, ground_truths, srcnames, h = loadData(dataname)
    
    
    
    #Run algorithm
    if args.truth_inference == "TDH":
        ti_model = TDH(claims, deepCopyClaimsByObj(claimsByObj),names, h, min_max_iter)
    elif args.truth_inference == "Vote":
        ti_model = Vote(claims, deepCopyClaimsByObj(claimsByObj),names, h, min_max_iter)
    ti_model.run(args.verbose)
    
    
    #evaluation
    evaluation = EvaluationModule(ground_truths, names, claimsByObj, h)
    accuracy, accuracy_gen, avg_distance = evaluation.eval(ti_model.getTruths())

    if args.crowdsourcing:
        workers = loadSimulatedWorkersUniform(0.7,0.8,num_workers,ground_truths)
        ti_model.regiWorkers(num_workers)
        for r in range(1,num_rounds+1):
            print("Round",r)
            if args.task_assignment == "ME":
                ti_model.assignTasksMaxEntropy(workers, numQuestions, 0, r)
            elif args.task_assignment == "EAI":
                ti_model.assignTasksMaxAccuracyIncrease(workers, numQuestions, 0, r)
            
            ti_model.regiAnswers(workers)
            
            ti_model.run(args.verbose)
            accuracy, accuracy_gen, avg_distance = evaluation.eval(ti_model.getTruths())
    
    
    
    