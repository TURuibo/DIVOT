# Structure of the directory

"pairs/""  
the causal-effect pari datasets, which can be downloaded at https://webdav.tuebingen.mpg.de/cause-effect/    

"results/"  
the results of DIVOT for causal direction determination  

"params.json"  
the json file with all the parameters for an experiment  

"run_realdata.py"  
the source code for the experiments  

"util.py"  
functions for DIVOT  


# How to run the experiments
To reproduce the real-world data experiments, one can run with the command in terminal:  

python3 run_realdata.py params data_filename1 data_filename2 ...  
params: the json file with all the parameters for an experiment  
data_filename: the file name of the dataset in the directory "pairs/"  

```
python3 run_realdata.py params 0001 0020 0048 0100
```

The results will be saved as a ".txt" file in the directory "results/".  
the first column: file name, e.g., 0001  
the second column: suppose that x causes y, the mean of the loss of batches , e.g., 0.18627807  
the third column: suppose that y causes x, the mean of the loss of batches , e.g., 0.19739667  
the fourth column: suppose that x causes y, the standard deviation of the loss of batches, e.g., 0.0034677496  
the fifth column: suppose that y causes x, the standard deviation of the loss of batches , e.g., 0.0018815454  

If the value in the second column is smaller than the value in the third column, then x causes y; otherwise, y causes x.  
