import datetime
import pygame
import threading


class AlarmManager:

    def __init__(self):
        print "init AlarmManager"
        self.__scoreAccept = 0.6
        self.__scoreReject = 0.4
        self.__washBasin = 'washBasin'
        self.__plate = 'plate'
        self.__spatula = 'spatula'
        pygame.mixer.init()

    def processAlarm(self, human_string, score):

        print "process alarm for", human_string, score

        if self.isActive():
            print "Alarm is already active. Aborting"
            return

        if self.__washBasin not in human_string:
            self.soundAlarmAsync()
            return

        if self.__washBasin in human_string and self.__isScoreRejected(score):
            self.soundAlarmAsync()
            return

        if self.__plate in human_string or self.__spatula in human_string:
            self.soundAlarmAsync()
            return

    def isWindowSlideNeeded(self, human_string, score):
        return self.__washBasin in human_string and \
               self.__isScoreAccepted(score) and \
               score <= 0.90

    def soundAlarmAsync(self):
        thread = threading.Thread(target=self.__soundAlarm, args=([]))
        thread.start()

    def __soundAlarm(self):

        for i in range(5):
            print("sounding alarm......................\n")
        self.stopAlarm()
        pygame.mixer.music.load("/Users/oofong/Projects/Hackathon/HackathonProject/kitchenbot/audio/siren.wav")
        pygame.mixer.music.play(1)  # play 1 extra time
        while pygame.mixer.music.get_busy():
            pygame.time.delay(100)
        print "__soundAlarm finished"

    def stopAlarm(self):
        print("stopping alarm......................\n")
        pygame.mixer.music.stop()

    def isActive(self):
        return pygame.mixer.music.get_busy()

    def __isScoreAccepted(self, score):
        return score >= self.__scoreAccept

    def __isScoreRejected(self, score):
        return score <= self.__scoreReject
