#number fo datasets
import pandas as pd
n = 4

x = "handtracking"+str(1)+".csv"

x = pd.read_csv(x)

for i in range(2,n+1):
        temp = "handtracking"+str(i)+".csv"
        temp = pd.read_csv(temp)
        x = pd.concat([x,temp])

x.to_csv("train.csv")
