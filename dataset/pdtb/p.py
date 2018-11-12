import json
from pprint import pprint
import numpy
import os
import codecs
import string
f1=open('./ffsss.txt', 'w')  
f2=open('./fflll.txt', 'w')  
#for fp in ['../wsj.txt' ,'../ap.txt' ,'../nyt.txt']:
#    with open(fp) as f:
#        for line in f:
#            ee=line.split('\t')
#            f1.write(ee[3][1:-1]+'\n')
#            if ee[7]=="gen":
#               f2.write('1\n')
#            elif ee[7]=="spec":
#                f2.write('2\n')
        
        
        
for filename in os.listdir('./'):
    if filename.endswith(".json"): 
        with open(filename) as data_file:    
            data = json.load(data_file)
            for i in range(len(data)):
                if len(data[i]['ratings'])>0:
                    if numpy.mean(data[i]['ratings'])>3.5:
                        f1.write(data[i]['tkn']+'\n')
                        f2.write('1\n')
                    elif numpy.mean(data[i]['ratings'])<=3.5:
                        f1.write(data[i]['tkn']+'\n')
                        f2.write('2\n')
                        
          