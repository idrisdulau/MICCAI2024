import os
import cv2
import sys
import tqdm
import numpy

def writeIMG(arr, writePath, imgName):
    os.makedirs(writePath, exist_ok=True)
    cv2.imwrite(os.path.join(writePath,imgName), arr)

def main(argv):
    assert(len(argv) == 4)
    veinsPath, arteriesPath, outputPath = argv[1:4]

    for imgName in tqdm.tqdm(sorted(os.listdir(veinsPath))):      
        veins = cv2.imread(os.path.join(veinsPath,imgName), cv2.IMREAD_UNCHANGED)
        arteries = cv2.imread(os.path.join(arteriesPath,imgName), cv2.IMREAD_UNCHANGED)
        
        green = veins & arteries
        blue = veins - green
        red = arteries - green

        avo = cv2.merge((blue,green,red))
        writeIMG(avo, outputPath, imgName)

if __name__ == '__main__':
    main(sys.argv)
