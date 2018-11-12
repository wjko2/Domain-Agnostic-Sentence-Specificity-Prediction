f=open('forexample.csv','r')
#f2=open('data.txt','r')
fw=open('fex.txt','w')
fw2=open('fexx.txt','w')
for line in f:
    line=line.split('|')
    fw.write(line[0].rstrip().lstrip()+'\n')
    fw.write(line[1].rstrip().lstrip()+'\n')
    fw2.write('1\n2\n')
        

