import os
import cv2
import sys
import tqdm
import numpy

def writeIMG(arr, writePath, imgName):
    os.makedirs(writePath, exist_ok=True)
    cv2.imwrite(os.path.join(writePath,imgName), arr)

def artifactRemoval(arr, threshold):
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr.astype(numpy.uint8), connectivity=8)
    sortedSequences = sorted([s[cv2.CC_STAT_AREA] for s in stats[1:]], reverse=True)
    tmpAreaList = sortedSequences[0:int(threshold)]
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] in tmpAreaList:
            tmpAreaList.remove(stats[i, cv2.CC_STAT_AREA])
        else:
            arr[labels == i] = 0
    return arr

def main(argv):
    assert(len(argv) == 5)
    inPath,outPathArteries,outPathVeins,mode = argv[1:5]
    assert(mode == "Basic" or mode == "CC1" or mode == "CCBest")

    os.makedirs(outPathArteries, exist_ok=True)
    os.makedirs(outPathVeins, exist_ok=True)

    for imgName in tqdm.tqdm(sorted(os.listdir(inPath))):      
        avo = cv2.imread(os.path.join(inPath,imgName.split(".")[0]+".png"), cv2.IMREAD_UNCHANGED)

        blue, green, red = cv2.split(avo)

        arteries = red | green
        veins  = blue | green

        if mode == "CC1":
            threshold = 1
            arteries = artifactRemoval(arteries, threshold)
            veins = artifactRemoval(veins, threshold)

        if mode == "CCBest":
            arteries = artifactRemoval(arteries, 12)
            veins = artifactRemoval(veins, 10)

        # cv2.imshow("concat", cv2.hconcat([arteries,veins]))
        # cv2.waitKey(0)
        # exit()

        writeIMG(arteries, outPathArteries,imgName)
        writeIMG(veins, outPathVeins,imgName)

if __name__ == '__main__':
    main(sys.argv)
