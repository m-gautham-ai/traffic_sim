import random
import time
import threading
import pygame
import sys
import os

# Default values of signal timers
defaultGreen = {0:10, 1:10, 2:10, 3:10}
defaultRed = 150
defaultYellow = 5

signals = []
noOfSignals = 4
currentGreen = 0   # Indicates which signal is green currently
nextGreen = (currentGreen+1)%noOfSignals    # Indicates which signal will turn green next
currentYellow = 0   # Indicates whether yellow signal is on or off 

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'bike':2.5}  # average speeds of vehicles

# Vehicle spawn areas and boundaries
x = {'right':0, 'down':(700, 750), 'left':1400, 'up':(600, 650)}
# y = {'right':(350, 400), 'down':0, 'left':(450, 500), 'up':800}
y = {'right':(350, 500), 'down':0, 'left':(450, 500), 'up':800}

vehicles = {'right': [], 'down': [], 'left': [], 'up': []}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'bike'}
directionNumbers = {0:'right'}

# Coordinates of signal image, timer, and vehicle count
# signalCoods = [(530,230),(810,230),(810,570),(530,570)]
# signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

# Gap between vehicles
stoppingGap = 25    # stopping gap
movingGap = 25   # moving gap

# set allowed vehicle types here
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
allowedVehicleTypesList = []
# set random or default green signal time here 
randomGreenSignalTimer = True
# set random green signal time range here 
randomGreenSignalTimerRange = [10,20]

timeElapsed = 0
simulationTime = 300
timeElapsedCoods = (1100,50)
vehicleCountTexts = ["0", "0", "0", "0"]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, currentSignal):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.currentSignal = currentSignal
        self.signalText = ""
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.crossed = 0

        # Randomize starting position
        if direction == 'right':
            self.x = x[direction]
            self.y = random.randint(y[direction][0], y[direction][1])
        elif direction == 'left':
            self.x = x[direction]
            self.y = random.randint(y[direction][0], y[direction][1])
        print(self.x, self.y)
        # elif direction == 'down':
        #     self.y = y[direction]
        #     self.x = random.randint(x[direction][0], x[direction][1])
        # elif direction == 'up':
        #     self.y = y[direction]
        #     self.x = random.randint(x[direction][0], x[direction][1])

        vehicles[direction].append(self)
        self.index = len(vehicles[direction]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.stop = defaultStop[direction]

        # Prevent spawn overlap
        if len(vehicles[direction]) > 1 and self.is_too_close(vehicles[direction][self.index-1]):
            if direction == 'right':
                self.x = vehicles[direction][self.index-1].x - self.image.get_rect().width - movingGap
            elif direction == 'left':
                self.x = vehicles[direction][self.index-1].x + self.image.get_rect().width + movingGap
            # elif direction == 'down':
            #     self.y = vehicles[direction][self.index-1].y - self.image.get_rect().height - movingGap
            # elif direction == 'up':
            #     self.y = vehicles[direction][self.index-1].y + self.image.get_rect().height + movingGap

        simulation.add(self)

    def is_too_close(self, other_vehicle):
        if self.direction == 'right' or self.direction == 'left':
            return abs(self.y - other_vehicle.y) < self.image.get_rect().height
        else:
            return abs(self.x - other_vehicle.x) < self.image.get_rect().width

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        # Check for vehicles ahead
        can_move = True
        for other_vehicle in vehicles[self.direction]:
            if self.index > other_vehicle.index: # only check vehicles ahead
                if self.direction == 'right' and self.x + self.image.get_rect().width > other_vehicle.x - movingGap and self.is_too_close(other_vehicle):
                    can_move = False
                    break
                elif self.direction == 'left' and self.x < other_vehicle.x + other_vehicle.image.get_rect().width + movingGap and self.is_too_close(other_vehicle):
                    can_move = False
                    break
                elif self.direction == 'down' and self.y + self.image.get_rect().height > other_vehicle.y - movingGap and self.is_too_close(other_vehicle):
                    can_move = False
                    break
                elif self.direction == 'up' and self.y < other_vehicle.y + other_vehicle.image.get_rect().height + movingGap and self.is_too_close(other_vehicle):
                    can_move = False
                    break

        # Obey traffic signal and update crossed status
        if self.crossed == 0:
            if self.direction == 'right' and self.x + self.image.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                if currentGreen != 0: can_move = False
            elif self.direction == 'left' and self.x < stopLines[self.direction]:
                self.crossed = 1
                if currentGreen != 2: can_move = False
            elif self.direction == 'down' and self.y + self.image.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                if currentGreen != 1: can_move = False
            elif self.direction == 'up' and self.y < stopLines[self.direction]:
                self.crossed = 1
                if currentGreen != 3: can_move = False

        if can_move:
            # Primary movement
            if self.direction == 'right':
                self.x += self.speed
            elif self.direction == 'left':
                self.x -= self.speed
            elif self.direction == 'down':
                self.y += self.speed
            elif self.direction == 'up':
                self.y -= self.speed

            # Lateral movement (drift)
            drift = random.uniform(-0.5, 0.5) * self.speed
            if self.direction == 'right' or self.direction == 'left':
                self.y += drift
                # Boundary check
                if self.y < y[self.direction][0]: self.y = y[self.direction][0]
                if self.y > y[self.direction][1]: self.y = y[self.direction][1]
            else: # up or down
                self.x += drift
                # Boundary check
                if self.x < x[self.direction][0]: self.x = x[self.direction][0]
                if self.x > x[self.direction][1]: self.x = x[self.direction][1] 

# Initialization of signals with default values
# def initialize(init_done_event):
#     minTime = randomGreenSignalTimerRange[0]
#     maxTime = randomGreenSignalTimerRange[1]
#     if(randomGreenSignalTimer):
#         ts1 = TrafficSignal(0, defaultYellow, random.randint(minTime,maxTime), "green")
#         signals.append(ts1)
#         ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, random.randint(minTime,maxTime), "red")
#         signals.append(ts2)
#         ts3 = TrafficSignal(defaultRed, defaultYellow, random.randint(minTime,maxTime), "red")
#         signals.append(ts3)
#         ts4 = TrafficSignal(defaultRed, defaultYellow, random.randint(minTime,maxTime), "red")
#         signals.append(ts4)
#     else:
#         ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0], "green")
#         signals.append(ts1)
#         ts2 = TrafficSignal(ts1.yellow+ts1.green, defaultYellow, defaultGreen[1], "red")
#         signals.append(ts2)
#         ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2], "red")
#         signals.append(ts3)
#         ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3], "red")
#         signals.append(ts4)
#     init_done_event.set()
#     repeat()

# Print the signal timers on cmd
# def printStatus():
#     for i in range(0, 4):
#         if(signals[i] != None):
#             if(i==currentGreen):
#                 if(currentYellow==0):
#                     print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
#                 else:
#                     print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
#             else:
#                 print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
#     print()  

def repeat():
    global currentGreen, currentYellow, nextGreen
    while(signals[currentGreen].green>0):   # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    # reset stop coordinates of vehicles 
    for vehicle in vehicles[directionNumbers[currentGreen]]:
        vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while(signals[currentGreen].yellow>0):  # while the timer of current yellow signal is not zero
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off
    
    # reset all signal times of current signal to default/random times
    if(randomGreenSignalTimer):
        signals[currentGreen].green = random.randint(randomGreenSignalTimerRange[0],randomGreenSignalTimerRange[1])
    else:
        signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = (currentGreen+1)%noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0): # current signal is green
                signals[i].green -= 1
                signals[i].currentSignal = "green"
            else: # current signal is yellow
                signals[i].yellow -= 1
                signals[i].currentSignal = "yellow"
        else:
            signals[i].red -= 1
            signals[i].currentSignal = "red"

# Generating vehicles in the simulation
def generateVehicles():
    while(True):
        vehicle_type = random.choice(allowedVehicleTypesList)
        direction_number = random.choice(list(directionNumbers.keys()))
        Vehicle(vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(random.randint(2, 5) / 10 ) 

# def showStats():
#     totalVehicles = 0
#     print('Direction-wise Vehicle Counts')
#     for i in directionNumbers.keys():
#         if(signals[i]!=None):
#             crossed_count = sum(1 for v in vehicles[directionNumbers[i]] if v.crossed == 1)
#             print("Direction ",i+1,": ", crossed_count)
#             totalVehicles += crossed_count
#     print('Total vehicles passed:',totalVehicles)
#     print('Total time:',timeElapsed)

def simTime():
    global timeElapsed, simulationTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simulationTime):
            showStats()
            os._exit(1) 

class Main:
    global allowedVehicleTypesList
    i = 0
    for vehicleType in allowedVehicleTypes:
        if(allowedVehicleTypes[vehicleType]):
            allowedVehicleTypesList.append(i)
            i += 1

class Main:
    running = True

    def __init__(self):
        # initialization_done = threading.Event()
        # thread1 = threading.Thread(name="initialization",target=initialize, args=(initialization_done,)) 
        # thread1.daemon = True
        # thread1.start()
        # initialization_done.wait()

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        self.screenWidth = 1400
        self.screenHeight = 800
        self.screenSize = (self.screenWidth, self.screenHeight)

        self.background = pygame.image.load('images/intersection.png')

        self.screen = pygame.display.set_mode(self.screenSize)
        pygame.display.set_caption("SIMULATION")

        self.redSignal = pygame.image.load('images/signals/red.png')
        self.yellowSignal = pygame.image.load('images/signals/yellow.png')
        self.greenSignal = pygame.image.load('images/signals/green.png')
        self.font = pygame.font.Font(None, 30)

        thread2 = threading.Thread(name="generateVehicles",target=generateVehicles, args=()) 
        thread2.daemon = True
        thread2.start()

        thread3 = threading.Thread(name="simTime",target=simTime, args=()) 
        thread3.daemon = True
        thread3.start()

        self.run()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    # showStats()
                    pygame.quit()
                    sys.exit()

            self.screen.blit(self.background,(0,0))   
            # for i in range(0,noOfSignals):
            #     if(signals[i].currentSignal=="green"):
            #         signals[i].signalText = signals[i].green
            #         self.screen.blit(self.greenSignal, signalCoods[i])
            #     elif(signals[i].currentSignal=="yellow"):
            #         signals[i].signalText = signals[i].yellow
            #         self.screen.blit(self.yellowSignal, signalCoods[i])
            #     else:
            #         signals[i].signalText = signals[i].red
            #         self.screen.blit(self.redSignal, signalCoods[i])
            
            # signalTexts = ["","","",""]
            # for i in range(0,noOfSignals):  
            #     signalTexts[i] = self.font.render(str(signals[i].signalText), True, self.white, self.black)
            #     self.screen.blit(signalTexts[i],signalTimerCoods[i])

            for i in directionNumbers.keys():
                crossed_count = sum(1 for v in vehicles[directionNumbers[i]] if v.crossed == 1)
                vehicleCountTexts[i] = self.font.render(str(crossed_count), True, self.black, self.white)
                self.screen.blit(vehicleCountTexts[i],vehicleCountCoods[i])

            timeElapsedText = self.font.render(("Time Elapsed: "+str(timeElapsed)), True, self.black, self.white)
            self.screen.blit(timeElapsedText,timeElapsedCoods)

            for vehicle in simulation:  
                self.screen.blit(vehicle.image, [vehicle.x, vehicle.y])
                vehicle.move()
            pygame.display.update()

Main()