import json
import csv
import argparse
import os
import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser(description="json argu", add_help=False)
    parser.add_argument("--jsonfilepath", type = str, default="(cifar100).json", help ='json file path for model result info.')
    parser.add_argument("--maincontent", type = str, default="Round Time Result Summary", help ='json file main content.')
    parser.add_argument("--savepath", type = str, default="./modelresultCSV/", help ='save csv path')
    parser.add_argument("--mainfunc", type = str, default="time", help ='set main function, all, time.')
    return parser.parse_args()
args =  get_arguments() 

class Jsonwriter(object):
    def __init__(self, args):
        self.args = args
        self.initrowinfo()
        
    def initrowinfo(self):
        self.epoch = 'Epoch'
        self.lr = 'Learning_Rate'
        self.trainlastacc = 'Train_Last_Acc'
        self.trainclassifieracc_1 = 'Train_Classifier_Acc_1'
        self.trainclassifieracc_2 = 'Train_Classifier_Acc_2'
        self.trainclassifieracc_3 = 'Train_Classifier_Acc_3'
        self.trainclassifieracc_4 = 'Train_Classifier_Acc_4'
        self.loss = 'Loss'
        self.traintime = 'Train_Time'
        self.testlastacc = 'Test_Last_Acc'
        self.testclassifieracc_1 = 'Test_Classifier_Acc_1'
        self.testclassifieracc_2 = 'Test_Classifier_Acc_2'
        self.testclassifieracc_3 = 'Test_Classifier_Acc_3'
        self.testclassifieracc_4 = 'Test_Classifier_Acc_4'
        self.testtime = 'Test_Time'
    
    def read_json_file(self, args):
        with open('./modelresult/' + args.jsonfilepath) as f:
            data = json.load(f)
        # print(data['Round Time Result Summary']['0']['1'])
        return data[args.maincontent]
    
    def json_info(self, jsoncontent):
        content = {
            self.epoch : jsoncontent['Epoch'],
            self.lr : jsoncontent['Learning Rate'],
            self.trainlastacc : jsoncontent['Train Last Acc'],
            self.trainclassifieracc_1 : jsoncontent['Train Classifier Acc'][0],
            self.trainclassifieracc_2 : jsoncontent['Train Classifier Acc'][1],
            self.trainclassifieracc_3 : jsoncontent['Train Classifier Acc'][2],
            self.trainclassifieracc_4 : jsoncontent['Train Classifier Acc'][3],
            self.loss : jsoncontent['Train Loss'],
            self.traintime : jsoncontent['Train Time'],
            self.testlastacc : jsoncontent['Test Last Acc'],
            self.testclassifieracc_1 : jsoncontent['Test Classifier Acc'][0],
            self.testclassifieracc_2 : jsoncontent['Test Classifier Acc'][1],
            self.testclassifieracc_3 : jsoncontent['Test Classifier Acc'][2],
            self.testclassifieracc_4 : jsoncontent['Test Classifier Acc'][3],
            self.testtime : jsoncontent['Test Time'],
        }
        return content



def main():
    writer = Jsonwriter(args)
    jsondata = writer.read_json_file(args)
    if(args.mainfunc == "all"):
        for times, time in enumerate(jsondata):
            modelinfolist = list()
            for _, epoch in enumerate(jsondata[time]):
                content = writer.json_info(jsondata[time][epoch])
                modelinfolist.append(content)
            model_pd = pd.DataFrame(modelinfolist)
            if not os.path.exists(args.savepath):
                os.makedirs(args.savepath)
            model_pd.to_csv(args.savepath + 'result' + str(times) + '.csv', index=False)
    elif(args.mainfunc == "time"):
        totaltime  = 0
        for times, time in enumerate(jsondata):
            for _, epoch in enumerate(jsondata[time]):
                totaltime += jsondata[time][epoch]['Test Time']
            break
            
        print("time: {:.2f}".format(totaltime / 400))
    
if __name__ == '__main__':
    main()