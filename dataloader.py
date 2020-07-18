import codecs

from _pytest.capture import unicode
import unicodecsv

from hierarchy import Hierarchy

def deepCopyClaimsByObj(claimsByObj):
    cbo_copy = [[] for _ in range(len(claimsByObj))]
    for e in range(len(claimsByObj)):
        for claim in claimsByObj[e]:
            cbo_copy[e].append(claim)

    return cbo_copy
def readFileUnicode(filepath):
    dat = []
    with codecs.open(filepath,'rb') as csvfile:
        cr = unicodecsv.reader(csvfile)
        
        for row in cr:
            urow = [unicode(v) for v in row]
            dat.append(urow)
    return dat

def loadDataUnicode(dirpath):
    claims = readFileUnicode(dirpath+"/claims.csv")
    gts = readFileUnicode(dirpath+"/groundtruths.csv")
    h = readFileUnicode(dirpath+"/hierarchy.csv")
    
    claims_out =[(obj, src, value) for obj, src, value in claims[1:]] 
    h_out = [(int(id),name, int(pid))for id,name, pid in h[1:]]
    gt_out = {obj:value.split(",") for obj, value in gts[1:]}
    
    
    return claims_out, gt_out, h_out

def reform_claims(claims_dat):
    srcids = {}
    nameids = {}
    names = []
    srcnames = []
    
    claims = []
    for objname, src, value in claims_dat:
        if src in srcids:
            sid = srcids[src]
        else:
            sid = len(srcnames)
            srcnames.append(src)
            srcids[src] = sid
            
        if objname in nameids:
            oid = nameids[objname]
        else:
            oid = len(names)
            names.append(objname)
            nameids[objname] = oid
        
        claims.append((oid,(sid,value)))
        
        
    return claims, names, srcnames

def loadData(dataname):
    dirpath = "data/"+dataname
    
    dat_claims, dat_gt, dat_h = loadDataUnicode(dirpath)
    
    #Reform claims
    claims, names, srcnames = reform_claims(dat_claims) 
    claimsByObj = recordsByObject(claims,len(names))
    
    
    #Build Hierarchy
    h = Hierarchy()
    h.build(dat_h)
    #h.printH()
    h.setMaxDepth()
    
    if "heritages" in dataname:
        h.preprocessed = True
    else:
        h.preprocessed = False
    
    print("#claims: %d, #entities: %d, claims/entities: %f"%(len(claims),len(names),float(len(claims))/len(names)))
    
    return claims, claimsByObj, names, dat_gt, srcnames, h


def recordsByObject(records, numObjects):
    records_o = [[] for o in range(numObjects)]
    for rec in records:
        records_o[rec[0]].append(rec[1])
    return records_o

def recordsBySource(records):
    records_s = [[]]
    for rec in records:
        s = rec[1][0]
        oid = rec[0]
        val = rec[1][1]
        while len(records_s) <= s:
            records_s.append([])
        records_s[s].append((oid,val))
    return records_s
    
if __name__=="__main__":
    loadData("heritages")