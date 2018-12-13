import pygame
import threading


class AlarmManager:

    def __init__(self):
        print("init AlarmManager")
        self.__scoreThreshold = 0.70
        self.__clean = "clean"
        self.__dirty = "dirty"
        pygame.mixer.init()

    def processAlarm(self, human_string, score):

        print("[AlarmManager]: process alarm for {} {}".format(human_string, score))

        if self.isActive():
            print("Alarm is already active. Abort")
            return

        if not self.__isScoreAccepted(score):
            print("[AlarmManager]: Not confident about this decision. Do nothing")
            return

        if self.__dirty == human_string:
            print("[AlarmManager]: Dirty sink. Sounding alarm...")
            self.soundAlarmAsync()
            return

        print("[AlarmManager]: Clean sink. No need to sound an alarm")

    def soundAlarmAsync(self):
        thread = threading.Thread(target=self.__soundAlarm, args=([]))
        thread.start()

    def __soundAlarm(self):

        for i in range(5):
            print("sounding alarm......................\n")
        pygame.mixer.music.load(
            "/Users/oofong/Projects/Hackathon/HackathonProject/old_deprecated_kitchenbot/audio/siren.wav")
        pygame.mixer.music.play(1)  # play 1 extra time
        while pygame.mixer.music.get_busy():
            pygame.time.delay(100)
        print("__soundAlarm finished")

    def stopAlarm(self):
        if self.isActive():
            print("stopping alarm......................\n")
            pygame.mixer.music.stop()

    def isActive(self):
        return pygame.mixer.music.get_busy()

    def __isScoreAccepted(self, score):
        return float(score) >= float(self.__scoreThreshold)

# AlarmManager().processAlarm("dirty", 0.9683294)
