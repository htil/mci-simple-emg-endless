from physiogo import PhysioGo
import numpy as np
import time
from physiolearn import PhysioLearn
from realtime.ioserver import IOServer
import random
from threading import Thread

socketServer = None

def refresh(app):
    global socketServer
    window_size = 2.0
    channels = [0]
    bands = app.getRecentAvgBandPowers(window_size, channels)
    socketServer.send('data', bands[0].tolist())
    if bands != None:
        label = app.model.predict([bands[0]])
        print("SENDING OVER: ")
        print(label)
        socketServer.send('prediction', label[0])

def startAnalysis():
    event_mapping = {'Rest': 100, 'Squeeze': 98}
    #learn = PhysioLearn(1, "emg", event_mapping, "data/events-gen-cleaned-eve.txt")
    learn = PhysioLearn("EMG_Test_15_16_41", 1, "emg", event_mapping)
    learn.readFile("data/EMG_Test_15_16_41.csv")
    learn.createEvents("data/bryan_events_march29.txt")
    #learn.plotEvents(2)
    dataset = learn.featureExtraction(4, [2.0], [0.4])
    print(learn.dataset)
    myModel = learn.train_knn(dataset, 2)
    learn.saveObject(myModel, "models/bryan_model.pkl")
    # learn.train_lda(dataset)
    # learn.train_regression(dataset)
    score = learn.testLocalModel("models/lda-emg-squeeze.pkl", dataset)
    print(score)

def dataCollection(msg, app):
    print("Data collection happening")
    print(msg)
    app.close()

def init():
    app = PhysioGo("EMG_Test", 'COM6', "ganglion", write_data=False) # create app
    app.addBasicText()
    plots = app.addLinePlot("line_series1", yMin=-app.yRange, yMax=app.yRange)
    app.loadModel("models/lda-emg-squeeze.pkl")
    app.setRefresh(refresh)
    app.setGUIVisibility(False) 
    # socketServer.runDaemon()
    app.start()

def main():
    global socketServer
    socketServer  = IOServer()
    socketServer.on("initialize", lambda x: init())
    socketServer.on("learn", lambda x: startAnalysis())
    # socketServer.on("startCollection", lambda x: dataCollection("bro", app))
    socketServer.run()
    # app.start()

if __name__ == "__main__":
    main()