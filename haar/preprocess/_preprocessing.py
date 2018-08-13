import os
import cv2

colors = ["Green", "Yellow", "Red", "GreenLeft", "YellowLeft", "RedLeft"]

def processFromDir(dataDir, outputDir, typing, dataFile):
    dirs = os.listdir(dataDir)

    #Gets rid of all non directories
    deleted = 0
    for i in range(len(dirs)):
        if "." in dirs[i - deleted]:
            del dirs[i-deleted]
            deleted += 1

    for d in dirs:
        annoDir = "orginalAnnotations/trafficLightBox"
        path = "{}/{}/{}".format(dataDir, d, annoDir)
        dataFiles = os.listdir(path)

        processData = {}

        print("Processing annotations in directory {} for {}".format(d, typing))

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
    '''this function will take in an annotations file and will return a dictionary with
    the given annotations
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
    dataFile = open("annotations.txt", 'w')
    classesFile = open("classes.txt", 'w')

    for col in colors:
        classesFile.write("{}\n".format(col))
        processFromDir(dataDir, outputDir, col, dataFile)

    dataFile.close()
    classesFile.close()

if __name__=="__main__":
    colorCycle("../data/dayTrain", ".")
