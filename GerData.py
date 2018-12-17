import numpy as np
import pandas as pd
import random
array=np.random.randint(0,100,(1000,20))
data=pd.DataFrame(data=array,columns=[str(random.randint(500,100000)).zfill(5) for _ in range(20)])
data.to_csv('data.csv',index=0)