#Basic Page Needs

from collections import namedtuple
import csv
import glob
import os
import sys
import pycrfsuite

#Global Objects

DialogUtterance = namedtuple("DialogUtterance", ("act_tag", "speaker", "pos", "text"))
PosTag = namedtuple("PosTag", ("token", "pos"))


trainDirectoryName=sys.argv[1]
testDirectoryName=sys.argv[2]
fileToBePrint=sys.argv[3]


def get_data(data_dir):
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for dialog_filename in dialog_filenames:
        yield get_utterances_from_filename(dialog_filename)


def get_utterances_from_filename(dialog_csv_filename):
    with open(dialog_csv_filename, "r") as dialog_csv_file:
        return get_utterances_from_file(dialog_csv_file)

def get_utterances_from_file(dialog_csv_file):
    reader = csv.DictReader(dialog_csv_file)
    return [_dict_to_dialog_utterance(du_dict) for du_dict in reader]


def _dict_to_dialog_utterance(du_dict):
    for k, v in du_dict.items():
        if len(v.strip()) == 0:
            du_dict[k] = None
    if du_dict["pos"]:
        du_dict["pos"] = [
            PosTag(*token_pos_pair.split("/"))
            for token_pos_pair in du_dict["pos"].split()]
    return DialogUtterance(**du_dict)


def getListOfLabel(utterances):
    getActTagList=[]
    for utter in utterances:
        if (utter.act_tag != None):  # Check if dialog_tag is present or else insert UNKNOWN tag
            getActTagList.append(utter.act_tag)
        else:
            getActTagList.append("other")
    return getActTagList


def returnExtractedFeatures(utterances):
    fileLevelList = []
    #speaker_list = []
    for i in range(0 , len(utterances)):
        utterLevelList=[]
        if (i == 0):
            utterLevelList.append("FIRST")
        else:
            if utterances[i-1].speaker != utterances[i].speaker:
                utterLevelList.append("CHANGE_SPEAKER")

        token_pos_list= utterances[i].pos
        if (token_pos_list != None):
            for token_pos in token_pos_list:
                token = token_pos.token
                token_string="TOKEN_" + token
                utterLevelList.append(token_string)
                pos = token_pos.pos
                pos_string = "POS_" + pos
                utterLevelList.append(pos_string)
        else:
            utterLevelList.append("")
        fileLevelList.append(utterLevelList)
    return fileLevelList

def getFileNamesInTest(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


trainFeaturedList=[returnExtractedFeatures(item) for item in list(get_data(trainDirectoryName))]
trainLabeledList=[getListOfLabel(item) for item in list(get_data(trainDirectoryName))]

#testInputFeature=list(get_data(testDirectoryName))
#testInputLabel=list(get_data(testDirectoryName))

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(trainFeaturedList, trainLabeledList):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 150,  # stop earlier

    'feature.possible_transitions': True
})

trainer.params()
trainer.train('outputGeneratedFile')
tagger = pycrfsuite.Tagger()
tagger.open('outputGeneratedFile')

#testFeaturedList = [tagger.tag(returnExtractedFeatures(a)) for a in testInputFeature]
testLabeledList = [getListOfLabel(item) for item in list(get_data(testDirectoryName))]

outputHandle=open(fileToBePrint,'w',encoding='latin1')

#PredictedListBy CRF Trainer

predictedList=[]

#Get The name of the files in the testData
all_files=getFileNamesInTest(testDirectoryName)

for file in all_files:
    getCurrentFileName = file.split('/')
    absoluteFileName = getCurrentFileName[len(getCurrentFileName)-1]

    outputHandle.write("Filename=\""+absoluteFileName+"\"")
    outputHandle.write("\n")
    temp = tagger.tag(returnExtractedFeatures(get_utterances_from_filename(file)))
    #predictedList.append(temp)
    for item in temp:
        outputHandle.write(str(item) + "\n")
    outputHandle.write("\n")


#Evaluate Accuracy
#print (predictedList)

#print (testLabeledList)
'''
correctLabel = 0
totalLabel = 0
for  i in range(0, len(predictedList)):
    totalLabel +=  len( predictedList[i])
    for j in range (0, len(predictedList[i])):
        if testLabeledList[i][j] == predictedList[i][j]:
            correctLabel += 1

accuracy = (correctLabel / totalLabel) * 100
print (accuracy)
'''


