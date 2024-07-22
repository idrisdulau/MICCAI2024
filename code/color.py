import os
import cv2
import sys
import tqdm
import numpy
import skimage

def statsCC(arr):
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(arr.astype(numpy.uint8), connectivity=8)
    return numLabels, labels, stats, centroids

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

def softVNR(arr, od, threshold):
    arr = arr | od
    arrDilat = cv2.dilate(arr.copy(), numpy.ones((11,11)))
    arrSK = skimage.morphology.skeletonize(arrDilat).astype(numpy.uint8)*255
    arrMerged = arr | arrSK
    arrMergedRemoved = artifactRemoval(arrMerged,threshold)
    arrFinal = arrMergedRemoved & arr
    return arrFinal

def baseMerge(arteries, veins, vessels):
    green = arteries & veins
    blue = veins - green
    red = arteries - green

    avoRGB = cv2.merge((blue,green,red))
    vesselsRGB = cv2.merge((vessels,vessels,vessels))

    mask = numpy.all(avoRGB == (0,0,0), axis = -1)
    avoBase = avoRGB.copy()
    avoBase[mask] = vesselsRGB[mask]

    return avoRGB, avoBase

def writeIMG(arr, writePath, imgName):
    os.makedirs(writePath, exist_ok=True)
    cv2.imwrite(os.path.join(writePath,imgName), arr)

def getContour(arr):
    dilat = cv2.dilate(arr.copy(), numpy.ones((3,3)))
    return dilat-arr

def colorize(subImg, avoBase):
    contour = getContour(subImg)
    contourRGB = cv2.merge((contour,contour,contour))
    # cv2.imshow("contourRGB",contourRGB.astype(numpy.uint8))

    mask = numpy.all(contourRGB == (255,255,255), axis=-1)
    neighborsImg = numpy.zeros_like(avoBase)
    neighborsImg[mask] = avoBase[mask]
    # cv2.imshow("neighborsImg",neighborsImg.astype(numpy.uint8))

    pos = numpy.argwhere(neighborsImg)
    dim = pos[:, 2]
    unique = numpy.unique(dim)   #Sorted ascending order   

    # print("sum:",sum(unique))
    # print("unique:",unique)

    def getChannel(unique):
        if len(unique)==1:
            if unique[0] == 0:
                # print("blue")
                return (255,0,0)
            elif unique[0] == 1:
                # print("green")
                return (0,255,0)
            else:
                # print("red")
                return (0,0,255)
        else:
            # print("black")
            return (0,0,0)

    subImgRGB = cv2.merge((subImg,subImg,subImg))
    subImgColorized = numpy.zeros_like(subImgRGB)
    mask = numpy.all(subImgRGB == (255,255,255), axis=-1)
    subImgColorized[mask] = getChannel(unique)

    # cv2.imshow("subImgRGB",subImgRGB.astype(numpy.uint8))

    return subImgColorized

def main(argv):
    assert(len(argv) == 7)
    vesselsPath, veinsPath, arteriesPath, odPath, writePath, version = argv[1:7]

    for imgName in tqdm.tqdm(sorted(os.listdir(arteriesPath))):      
        arteries = cv2.imread(os.path.join(arteriesPath,imgName), cv2.IMREAD_UNCHANGED)
        veins = cv2.imread(os.path.join(veinsPath,imgName), cv2.IMREAD_UNCHANGED)
        vessels = cv2.imread(os.path.join(vesselsPath,imgName), cv2.IMREAD_UNCHANGED)
        od = cv2.imread(os.path.join(odPath,imgName), cv2.IMREAD_UNCHANGED)

        # print(imgName)
        if version == "V1" or version == "V2":
            thld = 5
            arteries = softVNR(arteries, od, thld)
            veins = softVNR(veins, od, thld)
        elif version == "V3" or version == "V4": 
            arteries = arteries | od
            veins  = veins | od

        # cv2.imshow("arteries post",arteries)
        # cv2.imshow("veins post",veins)
        avoRGB,avoBase = baseMerge(arteries, veins, vessels)
        # cv2.imshow("avoRGB",avoRGB)
        # cv2.imshow("avoBase",avoBase)
        # cv2.waitKey(0)

        writeIMG(avoRGB, os.path.join(writePath,"avoRGB"), imgName)
        writeIMG(avoBase, os.path.join(writePath,"avoBase"), imgName)

        mask = numpy.all(avoBase == (255,255,255), axis=-1)
        avoWhite = numpy.zeros_like(avoBase)
        avoWhite[mask] = cv2.merge((vessels,vessels,vessels))[mask]
        # cv2.imshow("avoWhite",avoWhite.astype(numpy.uint8))
        # cv2.waitKey(0)

        colorized = numpy.zeros_like(avoWhite) #RGB
        avoWhite,_,_ = cv2.split(avoWhite)
        numLabels,labels,stats,centroids = statsCC(avoWhite)
        # print("imgName:",imgName,"| numLabels:",numLabels)

        for label in range (1,numLabels):
        # for label in range (1,30):
            # print("label",label)
            subImg = numpy.where(labels==label,255,0).astype(numpy.uint8)

            if version == "V1":
                #slow
                subImgColorized = colorize(subImg, avoBase)
                colorized = colorized | subImgColorized  
            elif version == "V2" or version == "V3" or version == "V4":
                #fast
                if numpy.count_nonzero(subImg) < 20: #15 to 20 times faster with this pre-deletion, but does not perform caliber widening
                    # rem += 1
                    continue
                else:
                    # kep += 1
                    subImgColorized = colorize(subImg, avoBase)
                    colorized = colorized | subImgColorized   

        # cv2.imshow("colorized",colorized.astype(numpy.uint8))
        final = avoRGB | colorized
        # cv2.imshow("final",final.astype(numpy.uint8))

        if version == "V4":
            blue,green,red = cv2.split(final)
            veins = blue | green
            arteries = red | green
            thld = 5
            veins = softVNR(veins, od, thld)
            arteries = softVNR(arteries, od, thld)
            green = arteries & veins
            blue = veins - green
            red = arteries - green
            final = cv2.merge((blue,green,red))

        writeIMG(colorized, os.path.join(writePath,"colorized"), imgName)
        writeIMG(final, os.path.join(writePath,"final"), imgName)

if __name__ == '__main__':
    main(sys.argv)