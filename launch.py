import os
import cv2
import subprocess

################################### --- COLOR --- ################################### 
version = "V1" # V2, V3 and V4 are 15 to 20 times faster, but do not perform caliber widening
writePath = os.path.join("steps")
os.makedirs(writePath, exist_ok=True)
subprocess.run([
    "python3", os.path.join("code","color.py"),\
        os.path.join("pred","vessels"),\
        os.path.join("pred","veins"),\
        os.path.join("pred","arteries"),\
        os.path.join("pred","od"),\
        writePath,\
        version
])

################################### --- SPLIT --- ################################### 
mode = "Basic"
subprocess.run(["python3", os.path.join("code","split.py"),\
    os.path.join("steps","final"),\
    os.path.join("steps","finalArteries"),\
    os.path.join("steps","finalVeins"),\
    mode
])

################################### --- GAP FILLING --- ################################### 
for target in ["finalVeins","finalArteries"]:
    writePath = os.path.join("steps",target+"GF")
    os.makedirs(writePath, exist_ok=True)
    subprocess.run([
        "python3", os.path.join("code","gapFillingAV.py"),\
            os.path.join("steps",target),\
            writePath
    ])

################################### --- VNR-AV --- ################################### 
for target in ["finalVeins","finalArteries"]:
    writePath = os.path.join("output",target)
    os.makedirs(writePath, exist_ok=True)
    subprocess.run([
        "python3", os.path.join("code","VNRAV.py"),\
            os.path.join("steps",target+"GF"),\
            os.path.join("pred","od"),\
            writePath
    ])

################################### --- MERGE --- ################################### 
writePath = os.path.join("output","finalRGB")
os.makedirs(writePath, exist_ok=True)
subprocess.run([
    "python3", os.path.join("code","merge.py"),\
        os.path.join("output","finalVeins"),\
        os.path.join("output","finalArteries"),\
        writePath
])

###################################
fundus = cv2.imread(os.path.join(os.path.join("pred","fundus"),"01.jpg"), cv2.IMREAD_UNCHANGED)
avo = cv2.imread(os.path.join(os.path.join("output","finalRGB"),"01.png"), cv2.IMREAD_UNCHANGED)
cv2.imshow("<-- fundus : finalRGB -->", cv2.hconcat([fundus,avo]))
cv2.waitKey(0)