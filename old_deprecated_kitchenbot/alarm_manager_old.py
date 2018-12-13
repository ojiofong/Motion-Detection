import datetime
import pygame
import threading


class AlarmManager:

    def __init__(self):
        print
        "init AlarmManager"
        self.__scoreAccept = 0.6
        self.__scoreReject = 0.2
        self.__washBasin = 'washbasin'
        self.__kitchenItems = ['plate', 'spatula', 'pot', 'bowl', 'ladle', 'measuring cup', 'cup',
                               'mug', 'coffee mug', 'tray']
        pygame.mixer.init()

    def processAlarm(self, human_string, score, isWindowSlide=False):

        print
        "process alarm for {} {} windowSlide:{}".format(human_string, score, isWindowSlide)

        if isWindowSlide:
            if self.isKitchenWareFound(human_string):
                self.soundAlarmAsync()
        print("[Alarm] window slide alarm proccessing only. Abort early")
        return

        # Else perform regular checks

        if self.isActive():
            print("Alarm is already active. Aborting")
            return

        if self.__washBasin not in human_string.split(','):
            print("[Alarm] washbasin not detected")
            self.soundAlarmAsync()
            return

        if self.__washBasin in human_string.split(',') and self.__isScoreRejected(score):
            print("[Alarm] washbasin rejected for low score")
            self.soundAlarmAsync()
            return

        if self.isKitchenWareFound(human_string):
            self.soundAlarmAsync()
            return

        print("[Alarm] washbasin is clean, no need to sound an alarm")

    def isKitchenWareFound(self, human_string):

        for item in self.__kitchenItems:
            if item in human_string.split(','):
                print("[Alarm] kitchen item found->", item)
                return True

        return False

    def isWindowSlideNeeded(self, human_string, score):
        if True:
            return True

        return self.__washBasin in human_string and \
               self.__isScoreAccepted(score) and \
               score <= 0.90

    def soundAlarmAsync(self):
        thread = threading.Thread(target=self.__soundAlarm, args=([]))
        thread.start()

    def __soundAlarm(self):

        for i in range(5):
            print("sounding alarm......................\n")
        # self.stopAlarm()
        # pygame.mixer.music.load("/Users/oofong/Projects/Hackathon/HackathonProject/old_deprecated_kitchenbot/audio/siren.wav")
        # pygame.mixer.music.play(1)  # play 1 extra time
        # while pygame.mixer.music.get_busy():
        #     pygame.time.delay(100)
        print("__soundAlarm finished")

    def stopAlarm(self):
        print("stopping alarm......................\n")
        pygame.mixer.music.stop()

    def isActive(self):
        return pygame.mixer.music.get_busy()

    def __isScoreAccepted(self, score):
        return score >= self.__scoreAccept

    def __isScoreRejected(self, score):
        return score <= self.__scoreReject
