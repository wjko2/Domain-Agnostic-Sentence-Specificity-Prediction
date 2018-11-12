f=open('label.txt','r')
f2=open('data.txt','r')
fw=open('labelb.txt','w')
fw2=open('datab.txt','w')
k=0
q=1
while k<2877:
    k=k+1
    p=int(float(f.readline()))
    
    if p==q:
        fw.write(str(p)+'\n')
        fw2.write(f2.readline().rstrip().lstrip()+'\n')
        q=3-q
    

