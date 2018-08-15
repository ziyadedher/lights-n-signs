import os
import cv2
import pickle

colors = ["Green", "Yellow", "Red", "GreenLeft", "YellowLeft", "RedLeft"]

'''
Final outputs:
    tempAnnotations
        -allImages.txt: file with the path of all images of interest
        -annotations.txt: file with annotations for each image in the following format:
            path class classIndex numberBoundingBoxes x1 y1 w1 h1 x2 y2 w2 h2 ...
        -classes.txt: file with all of the class names found in the annotations

    textAnnotations: each image file has a corresponding txt file with its annotations
        -dayTrain--00001.txt
        -dayTrain--00002.txt
        -dayTrain--00003.txt

        annotations come in the following format:
            class, x y w h

    dataPickle.pkl: pickle file with all annotations in an organized dictionary. It has the following structure:
        Dict[imageFilePath]:
            Dict[classes]:
                List[All annotations of this class in the image]:
                    List[x, y, w, h]
'''

def processFromDir(dataDir, typing, dataFile):
    '''Initially take the annotations from the original files for each of the given colors
    The colors (in the form of typing) can be any of the above.

    Datadir - directory where the data file is
    typing - color that you are processing for
    dataFile - final file that will contain all of the annotations
    '''
    dirs = list(filter(lambda x: os.path.isdir(os.path.join(dataDir, x)), os.listdir(dataDir)))
    print(dirs)

    for d in dirs:
        annoDir = "orginalAnnotations/trafficLightBox"
        path = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(path)

        processData = {}

        #processes and saves all annotation files
        for f in dataFiles:
            print(f)
            if (typing + ".txt").lower() in f.lower():
                print("This file meets typing specifications")
                processData = process(open("{}/{}".format(path, f)).read(), d)

        annoDir = "frames"
        mainPath = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(mainPath)

        #Checks to see if file exists and then classifies it as positive or negative data
        for f in dataFiles:
            f = f.replace("\n", "")

            #is picture
            if ((".png" not in f) and (".jpg" not in f)):
                continue

            index = int(f.split("--")[1].split(".")[0])
            val = processData.get(index, -1)
            print(f)
            filePath = "{}/{}".format(mainPath, f)

            if val != -1:
                dataFile.write("{} {} {} {}\n".format(os.path.abspath(filePath), typing, colors.index(typing), " ".join(val.split(" ")[1:])))


def process(fileText, d):
    '''this function will take in an original annotations file and will return a dictionary with
    the given annotations

    fileText - text from the original csv file
    d - directory where this annotation was taken from for naming purposes
    '''
    fileText = fileText.split("\n")
    data = {}

    for i in range(len(fileText)):
        line = fileText[i].replace("\t"," ").replace("  "," ").split(" ")

        if line[0] != "":
            index = int(line[0])
            textIndex = "0"*(5-len(str(index))) + str(str(index))
            num = int(line[1])
            data[index] = "{}/frames/{}--{}.png {}".format(d, d, textIndex, " ".join(line[1:]))

    return data

def colorCycle(dataDir, outputDir):
    '''This function simply cycles through the different colors and collects the
    original annotations

    dataDir - directory that contains the original data
    outputDir - directory where you want the annotations to be
    '''
    dataFile = open(os.path.join(outputDir, "annotations.txt"), 'w')
    classesFile = open(os.path.join(outputDir, "classes.txt"), 'w')

    for col in colors:
        classesFile.write("{} {}\n".format(col, colors.index(col)))
        processFromDir(dataDir, col, dataFile)

    dataFile.close()
    classesFile.close()

def getAllImages(dataDir, outputDir):
    '''
    This will create a file that contains the path to all of the images

    dataDir - directory that contains the original data
    outputDir - directory where you want the annotations to be
    '''
    dirs = list(filter(lambda x: os.path.isdir(os.path.join(dataDir, x)), os.listdir(dataDir)))
    dataFile = open(os.path.join(outputDir, "allImages.txt"),'w')

    for d in dirs:
        path = os.path.abspath(os.path.join(dataDir, d, "frames"))

        imgs = os.listdir(path)

        for img in imgs:
            dataFile.write("{}\n".format(os.path.join(path, img)))

    dataFile.close()

def setupDirs(outputDir):
    '''Creates the directories necessary for the Annotations. Deletes earlier dataFiles
    if they exist in the directory

    outputDir - directory where you want the directories to be created in
    '''

    try:
        os.mkdir(os.path.join(outputDir, "tempAnnotations"))
    except FileExistsError:
        for files in os.listdir(os.path.join(outputDir, "tempAnnotations")):
            os.remove(os.path.join(outputDir, "tempAnnotations", files))

    try:
        os.mkdir(os.path.join(outputDir, "textAnnotations"))
    except FileExistsError:
        for files in os.listdir(os.path.join(outputDir, "textAnnotations")):
            os.remove(os.path.join(outputDir, "textAnnotations", files))

def combineAnnotations(tempDir, outputDir):
    '''This function will use the intermediary annotations to create firstly a
    pickle file that contains all the annotations information in a dictionary.
    It also creates a new directory where there is a text file for each image file
    that contains the annotations for the corresponding image

    tempDir - directory where the temporary annotations are
    outputDir - directory where you want the final annotations to be
    '''

    print("combining annotations...")
    dataDict = {}

    f = open(os.path.join(tempDir,"allImages.txt"), "r").read().split("\n")

    for line in f:
        if line == "":
            continue

        dataDict[line] = {}

    f = open(os.path.join(tempDir,"annotations.txt"), "r").read().split("\n")

    for line in f:
        if line == "":
            continue

        line = line.split(' ')
        classification = line[1]
        classIndex = line[2]
        num = int(line[3])

        for i in range(num):
            dataVal = dataDict[line[0]].get(classification, [])
            dataVal.append(line[4 + 4*i:8 + 4*i])
            dataDict[line[0]][classification] = dataVal

    p = open("dataPickle.pkl", 'wb')
    pickle.dump(dataDict, p)

    for image in dataDict.keys():
        newPath = "{}.txt".format(image.split("/")[-1].split(".")[0])
        f = open(os.path.join(outputDir, "textAnnotations", newPath), 'w')
        for classification in dataDict[image].keys():
            f.write("{}".format(classification))
            for data in dataDict[image][classification]:
                f.write(",{}".format(" ".join(data)))

            f.write("\n")
        f.close()


if __name__=="__main__":
    setupDirs(".")
    colorCycle("../data/dayTrain", "tempAnnotations")
    getAllImages("../data/dayTrain", "tempAnnotations")
    combineAnnotations("tempAnnotations", ".")
