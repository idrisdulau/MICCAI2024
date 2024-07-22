import os
import sys
import cv2
import sys
import math
import tqdm
import numpy
import scipy
import skimage

def isBigRecursive(statsSkeleton, agregatedBiggestCCLabelsList, label):

    a = [statsSkeleton[e, cv2.CC_STAT_AREA] for e in agregatedBiggestCCLabelsList]
    # print(a)
    biggestCCArea = sum(a)
    # print("biggestCCArea:",biggestCCArea)
    minCCArea = statsSkeleton[label, cv2.CC_STAT_AREA]  
    # print("rate:",biggestCCArea/minCCArea,"|","biggestCCArea:",biggestCCArea,"minCCArea:",minCCArea)
    return minCCArea > biggestCCArea/3

def isBig(statsSkeleton, labelValBiggestCC, label):
    biggestCCArea = statsSkeleton[labelValBiggestCC, cv2.CC_STAT_AREA]  
    minCCArea = statsSkeleton[label, cv2.CC_STAT_AREA]  
    # print("rate:",biggestCCArea/minCCArea,"|","biggestCCArea:",biggestCCArea,"minCCArea:",minCCArea)
    return minCCArea > biggestCCArea/3

def isCloseEnough(branchEndP, discP):
    # Get the closestEndPToOD in the branch
    distances = scipy.spatial.distance.cdist(branchEndP, [discP], "euclidean")
    # print(numpy.min(distances))
    return numpy.min(distances) < 200 #Experimentally observed

def isFarEnough(branchEndP, discP):
    # Get the closestEndPToOD in the branch
    distances = scipy.spatial.distance.cdist(branchEndP, [discP], "euclidean")
    # print(numpy.min(distances))
    return numpy.min(distances) > 400 #Experimentally observed 

def isCloseEnoughToBorder(img,branchEndP):
    lx,ly = [x for x,y in branchEndP],[y for x,y in branchEndP]
    # print(branchEndP)
    # print(lx, numpy.min(lx), numpy.max(lx))
    # print(ly, numpy.min(ly), numpy.max(ly))
    h,w = img.shape

    # Assuming that the images are correctly cropped, and that unnecessary padding is not added. 
    dist = 75
    return numpy.min(lx) < dist or numpy.min(ly) < dist or numpy.max(lx) > w-dist or numpy.max(ly) > h-dist

def getPListBelongsToLine(img, nearestEndCrossP, endP):
    x1, y1 = nearestEndCrossP
    x2, y2 = endP
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy
    
    linePList = []
    while True:
        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x2 += sx

        if e2 < dx:
            err += dx
            y2 += sy
        
        if x2 < 0 or x2 > img.shape[0]-1 or y2 < 0 or y2 > img.shape[1]-1:
            break

        linePList.append((x2, y2))
    return linePList

def getPListInsideLine3x3(skeleton, p1, p2):
    skeletonCopy = numpy.copy(skeleton).astype(numpy.uint8)
    black = numpy.copy(skeleton).astype(numpy.uint8)*0
    x1, y1 = p1
    x2, y2 = p2
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy
    
    linePList = []
    while True:
        x1 += sx if x1 != x2 else 0
        y1 += sy if y1 != y2 else 0
        if (x1, y1) == p2:
            break
        linePList.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
        if e2 < dx:
            err += dx
    
    for p in linePList:
        black[p] = 1
    # print(numpy.count_nonzero(black))

    kernel = numpy.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    black = scipy.signal.convolve2d(black, kernel, mode='same', boundary='fill', fillvalue=0)
    black = numpy.where(black != 0, 1, 0)
    black[p1] = 0
    black[p2] = 0  

    # cv2.imshow("skeletonCopy",skeletonCopy.astype(numpy.uint8)*150)
    # skeletonCopy[p1] = 250
    # skeletonCopy[p2] = 250
    # skeletonCopy = numpy.where(black !=0, black, skeletonCopy)
    # cv2.imshow("skeleton + black",skeletonCopy.astype(numpy.uint8)*150)

    # black = numpy.where(black != 0, 1, 0)
    # cv2.imshow("black",black.astype(numpy.uint8)*150)
    # cv2.waitKey(0)

    return black

def getClosestCrossingPointOfLine(skeleton, p1, p2):
    imgP = getPListInsideLine3x3(skeleton, p1, p2)
    merged = imgP & skeleton
    crossingPList = numpy.argwhere(merged==1)
    distances = scipy.spatial.distance.cdist(crossingPList, [p1], "euclidean")
    closestCrossingP = crossingPList[numpy.argmin(distances)]  

    # a = numpy.copy(skeleton).astype(numpy.uint8)*150
    # b = numpy.copy(imgP).astype(numpy.uint8)*255
    # cv2.imshow("skeleton | imgP", a | b)
    # cv2.waitKey(0)

    return closestCrossingP

def isCrossing(skeleton, p1, p2):
    imgP = getPListInsideLine3x3(skeleton, p1, p2)
    merged = imgP & skeleton
    return numpy.sum(merged) > 0

def recoBigAndClose(branchEndP, endPointsList, labels, labelValBiggestCC, discP, skeleton):
    # Get the closestEndPToOD in the branch
    distances = scipy.spatial.distance.cdist(branchEndP, [discP], "euclidean")
    closestEndPToOD = branchEndP[numpy.argmin(distances)]  
    distClosestEndPToOD = numpy.min(distances)

    # Get the closestEndPFromOD to the closest point that is either a mainEndP or the discCenter.      
    biggestBranchPix = numpy.argwhere(labels == labelValBiggestCC) 
    biggestBranchEndP = [(a[0],a[1]) for a in biggestBranchPix if any((a == b).all() for b in endPointsList)]

    distances = scipy.spatial.distance.cdist(biggestBranchEndP, [closestEndPToOD], "euclidean")
    closestEndPToBiggest = biggestBranchEndP[numpy.argmin(distances)]  
    distclosestEndPToBiggest = numpy.min(distances)

    # print(discP)
    # print(closestEndPToOD)
    # print(closestEndPToBiggest)

    # If it's shorter to reconnect to endPofBiggest
    if distclosestEndPToBiggest < distClosestEndPToOD:
        # if it cross when going to endPofBiggest, then go to disc
        if isCrossing(skeleton, closestEndPToOD, closestEndPToBiggest):
            # if it cross when going to disc, reco at closestsCrossingPoint
            if isCrossing(skeleton, closestEndPToOD, discP):
                closestCrossingP = getClosestCrossingPointOfLine(skeleton, closestEndPToOD, discP)
                # return (closestEndPToOD[1],closestEndPToOD[0]),(closestCrossingP[1],closestCrossingP[0])
                return (closestEndPToOD,closestCrossingP)
            # if it doesn't cross when going to disc, reco at disc
            else:
                # return (closestEndPToOD[1],closestEndPToOD[0]),(discP[1],discP[0])
                return (closestEndPToOD,discP)
        # if it doesn't cross when going to endPofBiggest, reco at endPofBiggest
        else:
            # return (closestEndPToOD[1],closestEndPToOD[0]),(closestEndPToBiggest[1],closestEndPToBiggest[0])
            return (closestEndPToOD,closestEndPToBiggest)
    # If it's shorter to reconnect to disc, reco at disc
    else:
        # return (closestEndPToOD[1],closestEndPToOD[0]),(discP[1],discP[0])   
        return (closestEndPToOD,discP)   

def getOdCenter(od):
    _, _, _, centroids = cv2.connectedComponentsWithStats(od, connectivity=8)
    x,y = centroids[1]
    discP = round(y),round(x) 
    return discP #y,x format

def isValidPath(img, endP, crossP):
    # print("isvalid", endP, crossP)
    imgCopy = numpy.copy(img)
    skeleton = skimage.morphology.skeletonize(imgCopy)
    top, bot = min(endP[0],crossP[0]),max(endP[0],crossP[0])
    left, right = min(endP[1],crossP[1]),max(endP[1],crossP[1])
    # The window is tight thus it will miss the count on the tortuous vessels but we cannot reconnect nicely in that case so it is good.
    pad = 9
    subRectSkeleton = skeleton[top-pad:bot+1+pad,left-pad:right+1+pad]
    skeletonSubrectArea = numpy.sum(subRectSkeleton)
    chebyshevDist = scipy.spatial.distance.chebyshev(endP,crossP) + pad

    # print("subRect:",skeletonSubrectArea,"|",
    #       "chebyshev:", chebyshevDist,"|",
    #       "moreArea:", skeletonSubrectArea > chebyshevDist,"|",
    #     #   "CC:",numLabels-1
    #       )
    # cv2.imshow("subRectSkeleton",subRectSkeleton.astype(numpy.uint8)*255)
    # cv2.waitKey(0)

    return skeletonSubrectArea > chebyshevDist

def propagateBranchWidth(img, startP, kernelSize=2):
    y, x = startP
    imgCopy = numpy.copy(img)
    imgPadded = numpy.pad(imgCopy, pad_width=kernelSize, mode='constant', constant_values=0)
    y += kernelSize
    x += kernelSize
    pattern = imgPadded[x-kernelSize : x+kernelSize+1, y-kernelSize : y+kernelSize+1] 
    pattern = numpy.where(pattern == 0, pattern, 255)

    rows, cols = numpy.where(imgPadded == 200)
    for r, c in zip(rows, cols):
        rMin = max(r-kernelSize, 0)
        rMax = min(r+kernelSize+1, imgPadded.shape[0])
        cMin = max(c-kernelSize, 0)
        cMax = min(c+kernelSize+1, imgPadded.shape[1])

        mask = imgPadded[rMin:rMax, cMin:cMax] != 255
        imgPadded[rMin:rMax, cMin:cMax][mask] = pattern[mask]
    
    imgUnPadded = imgPadded[kernelSize:-kernelSize, kernelSize:-kernelSize]
    imgUnPadded = imgUnPadded.astype(numpy.uint8)
    return imgUnPadded

def rebranchOrRemove(img, od, imgName):
    discP = getOdCenter(od)
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(img.astype(numpy.uint8), connectivity=8)
    labelValBiggestCC = numpy.argmax(stats[1:, cv2.CC_STAT_AREA])+1   

    labelAreas = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, numLabels)]
    sortedLabelsAreas = sorted(labelAreas, key=lambda x: x[1], reverse=True)
    sortedLabels = [label for (label,area) in sortedLabelsAreas]
    assert(sortedLabels[0] == labelValBiggestCC)

    skeleton = skimage.morphology.skeletonize(img) 
    skeletonCopy = numpy.copy(skeleton) 
    skeletonCopy = skeletonCopy.astype(numpy.uint8)*255
    kernel = numpy.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    countNeighborsOfAllPix = scipy.signal.convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    countNeighborsOfOnesPix = numpy.where(skeleton==1, countNeighborsOfAllPix, 0)
    endPointsList = numpy.argwhere(countNeighborsOfOnesPix==1) #y,x format
    crossPointsList = numpy.argwhere(countNeighborsOfOnesPix==3) #y,x format

    numLabelsSkeleton, labelsSkeleton, statsSkeleton, _ = cv2.connectedComponentsWithStats(skeleton.astype(numpy.uint8), connectivity=8)
    assert(numLabels == numLabelsSkeleton)

    toRebranchList = []
    toRemoveList = []
    sortedLabels.remove(labelValBiggestCC)

    agregatedBiggestCCLabelsList = [labelValBiggestCC]

    for label in sortedLabels:
        lendP = [endP for ((endP,newP)) in toRebranchList]
        agregatedBiggestCCLabelsList = [labelValBiggestCC]+[labels[e] for e in lendP]

        branchPix = numpy.argwhere(labels == label) 
        branchEndP = [(a[0],a[1]) for a in branchPix if any((a == b).all() for b in endPointsList)]
        branchcrossP = [(a[0],a[1]) for a in branchPix if any((a == b).all() for b in crossPointsList)]  

        # If the second biggest is very big, ensure a Reco near the OD. 
        # From the closest endP to the OD in the branch ==> to the closest point that is either a mainEndP or the discCenter.
        labelStatus = ""
        if isBigRecursive(statsSkeleton, agregatedBiggestCCLabelsList, label) and isCloseEnough(branchEndP, discP):

            p1, p2 = recoBigAndClose(branchEndP, endPointsList, labels, labelValBiggestCC, discP, skeleton)
            toRebranchList.append((p1, p2))
            candidatesToRebranchList = [] #Because toRebranchList is already filled.
            labelStatus = "big" #Avoid checking again the condition afterwards

        # If the CC is huge and far from disc at reco point, then keep it, don't even remove, just keep as a disconected FOV CC
        elif len(branchEndP)>0 and len(branchPix)>300 and isFarEnough(branchEndP, discP) and isCloseEnoughToBorder(img,branchEndP):
            # Neither rebranch nor remove
            candidatesToRebranchList = []
            labelStatus = "outside" #Avoid checking again the condition afterwards            

        else: 
            candidatesToRebranchList = []
            for endP in branchEndP:
                if len(branchPix) <= 2:
                    continue               
                               
                if branchcrossP == []:
                    branchEndPCopy = branchEndP.copy()
                    branchEndPCopy.remove(endP)
                    pList = getPListBelongsToLine(img, branchEndPCopy[0], endP)
                else:
                    nearestEndCrossP = -1 
                    distances = scipy.spatial.distance.cdist([endP], branchcrossP, "euclidean")
                    sortedIdx = numpy.argsort(distances)
                    for idx in sortedIdx[0]:
                        candidate = branchcrossP[idx]
                        if isValidPath(img, endP, candidate):
                            nearestEndCrossP = candidate
                            break  # As soon as a valid path is found
                    if nearestEndCrossP == -1:
                        pList = []
                    else: 
                        pList = getPListBelongsToLine(img, nearestEndCrossP, endP)

                for p in pList:
                    if img[p] == 255 and (labels[p] in agregatedBiggestCCLabelsList):  
                        endPtoMainPLength = scipy.spatial.distance.cdist([endP], [p], "euclidean")

                        #OD: Less incoherent paths and structures, but, less vascular information
                        mainPtoDiscPLength = scipy.spatial.distance.cdist([p], [discP], "euclidean")
                        candidatesToRebranchList.append((endP,p,endPtoMainPLength[0][0]+mainPtoDiscPLength[0][0]))   

                        #Classic: More vascular information, but, more incoherent paths and structures
                        # candidatesToRebranchList.append((endP,p,endPtoMainPLength[0][0]))         

        if len(candidatesToRebranchList)>0:
            sortedcandidatesToRebranchList = sorted(candidatesToRebranchList, key=lambda x: x[2]) #to rebranch the first valid in SHORTEST order
            endP, newP, pathLength = sortedcandidatesToRebranchList[0]
            minCCArea = statsSkeleton[labelsSkeleton[endP], cv2.CC_STAT_AREA]  

            acceptedRecoDistanceRate = 2
            if (pathLength < minCCArea/acceptedRecoDistanceRate):
                toRebranchList.append((endP, newP))
                
            else:
                toRemoveList.append(branchPix)

        elif labelStatus == "big" or labelStatus == "outside":
            pass
        else:
            toRemoveList.append(branchPix)

    for b in toRemoveList:
        for (y,x) in zip(b[:,0], b[:,1]):
            img[y, x] = 0

    for endP, newP in toRebranchList:
        cv2.line(img, endP[::-1], newP[::-1], 200, 1)
        img = propagateBranchWidth(img, endP[::-1])  
    
    return img

def main(argv):
    inputPath, odPath, outputPath = argv[1:4]
    for imgName in tqdm.tqdm(os.listdir(inputPath)):
        od = cv2.imread(os.path.join(odPath,imgName), cv2.IMREAD_UNCHANGED)
        img = cv2.imread(os.path.join(inputPath,imgName), cv2.IMREAD_UNCHANGED)
        img = rebranchOrRemove(img,od,imgName)
        cv2.imwrite(os.path.join(outputPath,imgName), img) 

if __name__ == '__main__':
    main(sys.argv)
