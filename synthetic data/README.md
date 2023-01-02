# Synthetic data experiments
## Simulation study in App. B.1.
"Submission fig1 ot vs fcm.html" is the Jupytor notebook for the simulation study.  


## Experiments in Sec. 6.1: Synthetic data experiments
"syn_exp1.html" is a Jupytor notebook of a demo for the experiments in Sec. 6.1. To reproduce the experiments in Sec. 6.1, one can first go the directory of the project and then use the python file "syn_exp1.py", e.g., the command in terminal  
python3 syn_exp1.py PARAM1 PARAM2 PARAM3 PARAM4  
PARAM1: number_of_experiments  
PARAM2: sample_size  
PARAM3: batch_size  
PARAM4: number_of_positions  

`python3 syn_exp1.py 100 10 0.4 10`  
`python3 syn_exp1.py 100 25 0.2 25`  
`python3 syn_exp1.py 100 50 0.2 50`  
`python3 syn_exp1.py 100 100 0.15 50`  
`python3 syn_exp1.py 100 200 0.15 50`  
`python3 syn_exp1.py 100 500 0.05 50`  

The default function is linear. By modifying the function **f_t(x)**, one can test ANMs with other nonlinear functions.  


## Experiments in App. B.2.3: Batches for the finite (or limited) sample scenario
"syn_exp2.html" is a Jupytor notebook of a demo for the experiments in App. B.2.3. To reproduce the experimental results in Fig. 5, one can first go the directory of the project and then run the the command in terminal,  
python3 syn_exp2.py PARAM1 PARAM2 PARAM3 PARAM4 PARAM5 PARAM6 PARAM7  
PARAM1: number of experiments  
PARAM2: sample size  
PARAM3: batch size  
PARAM4: number of positions   
PARAM5: number of updates(optimization)   
PARAM6: step size of the linear debiasing function parameter (optimization)  
PARAM6: step size of the noise distribution parameter (optimization)  

`python3 syn_exp2.py 100 100 0.05 100 500 8.0 1.0`
`python3 syn_exp2.py 100 100 0.1 100 500 8.0 1.0`
`python3 syn_exp2.py 100 100 0.15 100 500 8.0 1.0`
`python3 syn_exp2.py 100 100 0.2 100 500 8.0 1.0`
`python3 syn_exp2.py 100 100 0.3 100 300 1.0 1.0`
`python3 syn_exp2.py 100 100 0.5 100 300 1.0 1.0`
`python3 syn_exp2.py 100 100 0.6 100 300 1.0 1.0`
`python3 syn_exp2.py 100 100 0.7 100 100 1.0 1.0`
`python3 syn_exp2.py 100 100 0.8 100 100 1.0 1.0`
`python3 syn_exp2.py 100 100 0.9 100 100 1.0 1.0`
`python3 syn_exp2.py 100 100 0.99 100 100 1.0 1.0`


## Experiments in App. B.2.4: Debiasing functions for the few-sample scenario
"syn_exp3.html" is the Jupytor notebook which is used for the experimental result in Fig. 6.  

## Experiments in App. B.2.7: Robustness to prior misspecification
To reproduce the results in Fig. 7, one can first go the directory of the project and then run the the command in terminal,  
python3 syn_exp_rob.py PARAM1 PARAM2 PARAM3 PARAM4  
PARAM1: number_of_experiments  
PARAM2: sample_size  
PARAM3: batch_size  
PARAM4: number_of_positions  

`python3 syn_exp_rob.py 100 10 0.4 10`  
`python3 syn_exp_rob.py 100 25 0.2 25`  
`python3 syn_exp_rob.py 100 50 0.2 50`  
`python3 syn_exp_rob.py 100 100 0.15 50`  
`python3 syn_exp_rob.py 100 200 0.15 50`  
`python3 syn_exp_rob.py 100 500 0.05 50`  

The default function is linear. By modifying the function **f_t(x)**, one can test ANMs with other nonlinear functions. The default hypothesized noise distributino is beta distribution. By modifying distribution **random.beta(...)** in the function **test(...)**, one can use different hypothesized noise distributions for causal direction determination.  

## Experiments in App. B.2.8: Efficiency of DIVOT
"syn_exp5_eff.html" is the Jupytor notebook which is used for the experimental results in Table 2.   

## Experiments in App. B.2.9: Comparison with results of benchmark methods
"syn_exp4.html" is a Jupytor notebook of a demo for the experiments. One can reproduce the experimental results in Fig. 8 with  
python3 syn_exp_lap.py PARAM1 PARAM2 PARAM3 PARAM4  
PARAM1: number_of_experiments  
PARAM2: sample_size  
PARAM3: batch_size  
PARAM4: number_of_positions  

`python3 syn_exp_lap.py 100 25 0.2 25`  
`python3 syn_exp_lap.py 100 50 0.2 50`  
`python3 syn_exp_lap.py 100 75 0.2 10`  
`python3 syn_exp_lap.py 100 100 0.15 50`  
`python3 syn_exp_lap.py 100 200 0.15 50`  
`python3 syn_exp_lap.py 100 500 0.05 50`  

The default function is linear. By modifying the function **f_t(x)**, one can test ANMs with other nonlinear functions.  







