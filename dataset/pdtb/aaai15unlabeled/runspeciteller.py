
# files = ["apw.20000","nyt.20000","wsj.20000"]

# for f in files:
#     fin = open(f)
#     fout = open(f+".sents", "w")
#     for line in fin:
#         fout.write(line.strip().split("\t",2)[-1]+"\n")
#     fin.close()
#     fout.close()

cotrainfinalfile = "/home1/l/ljunyi/Projects/generalspecific/results/34000.newlab"
specitellerfile = "all.60000.spec"

train_preds = {}
with open(cotrainfinalfile) as f:
    for line in f:
        newlabid, p = line.strip().split()
        train_preds[int(newlabid)] = [int(p)]
with open(specitellerfile) as f:
    n = 0
    for line in f:
        p = float(line.strip())
        new_pred = -1 if p >= 0.5 else 1
        if n in train_preds:
            train_preds[n].append(new_pred)
        n += 1

allp = train_preds.values()
p, y = zip(*allp)

from util.evalclassifier import aprf
aprf(p,y)


