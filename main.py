import queue
import threading
import time
from random import randint
import random
import numpy as np
import pygame
import pygame_gui
from pygame_gui.elements import UITextEntryLine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def GenPop(count, len):
    inipop = []
    for i in range(count):
        inipop.append(random.sample(range(0, len), len))
    return inipop

def fitness(inipop):
    ft = []
    for x in inipop:
        penalty = 0
        for y in range(0, len(x)):
            for k in range(0, len(x)):
                if k != y:
                    if x[y] - y == x[k] - k or x[y] + y == x[k] + k:
                        penalty += 1
        ft.append(penalty)
    return ft

def returnfitness(list):
        penalty = 0
        for y in range(0, len(list)):
            for k in range(0, len(list)):
                if k != y:
                    if list[y] - y == list[k] - k or list[y] + y == list[k] + k or list[y] == list[k]:
                        penalty += 1
        return penalty

def Selection(graded):
    indexes = random.sample(range(0,len(graded)),3)
    return indexes[0],indexes[1],indexes[2]

def CrossOver(coprob,numberofGens,n,graded,weight):
    SolutionList = set()
    for iterations in range(0,numberofGens):
        graded = [(x[0], x[1]) for x in sorted(graded, reverse=False)]
        counter = 0
        for x in graded:
            if returnfitness(x[1]) == 0:
                SolutionList.add(tuple(x[1]))
                counter += 1
            else:
                ip1, ip2, ip3 = Selection(graded)
                while x[1] == graded[ip1][1] or x[1] == graded[ip2][1] or x[1] == graded[ip3][1]:
                    ip1, ip2, ip3 = Selection(graded)
                mutant = np.absolute(np.subtract(graded[ip1][1], graded[ip2][1]))
                for f in range(0,len(mutant)):
                    mutant[f] = int(mutant[f] * weight)
                mutant = np.add(mutant, graded[ip3][1])
                for k in range(0, len(mutant)):
                    if mutant[k] > n-1:
                        mutant[k] = mutant[k] % n
                Child = []
                Crossoverpoint = randint(0, 2)
                Startingpos = 0
                if (Crossoverpoint == 0):
                    Startingpos = 0
                else:
                    if Crossoverpoint == 1:
                        Child.append(mutant[0])
                        Startingpos = 1
                    else:
                        Child.append(mutant[0])
                        Child.append(mutant[1])
                        Startingpos = 2
                index = Startingpos
                while len(Child) != n:
                    rprob = random.uniform(0, 1)
                    if ( rprob < coprob) and mutant[index] not in Child:
                        Child.append(mutant[index])
                    elif x[1][index] not in Child and (rprob > coprob):
                        Child.append(x[1][index])
                    else:
                        r = randint(0,n-1)
                        while r in Child:
                            r = randint(0,n-1)
                        Child.append(r)
                    index = (index + 1) % n
                Childfit = returnfitness(Child)
                if Childfit == 0:
                    SolutionList.add(tuple(Child))
                if Childfit < x[0]:
                    graded.pop(counter)
                    graded.insert(counter, (Childfit, Child))
                counter += 1
    return graded,SolutionList


def StartAlgo(n,pop,Gen,probability,Weight,OGQ,ResQ): ## START Differential Algorithm actually solves the problem
    inipop = GenPop(pop,n)
    inipopClone = list(inipop)
    OGQ.put(inipopClone)
    ft = fitness(inipop)
    OGQ.put(ft)
    graded = [(x, inipop.pop(0)) for x in ft]
    graded,Solution = CrossOver(probability,Gen,n,graded,Weight)
    ResQ.put(graded)
    return Solution

def StartAlgoinThread(GeneticBoard,n1,n2,n3,IMAGE,Q,MAXQueue,SolutionQueue,EQ,OGQ,RSQ,AllSol): ## DIFFERENTIAl THREAD // calls StartAlgo
    Solutions = []
    Solutions = list(StartAlgo(n1, n2, n3, 0.7, 0.8,OGQ,RSQ))
    print("Gen Solutions : ",Solutions)
    print("Gen Solutionns LEN : ",len(Solutions))
    if len(Solutions) > 0 :
        GeneticBoard = CreateBoard(n1, 400, Solutions[0], IMAGE)
        EQ.put(time.thread_time())
        MAXQueue.put(len(Solutions))
        SolutionQueue.put(Solutions)
        Q.put(GeneticBoard)
        AllSol.append((len(Solutions),time.thread_time()))

#BACKTRACKING
solutionsBACK=[]
def backtracking(X):
    n=X
    board=[]
    for i in range(n):
        List = []
        for j in range(n):
            List.append(0)
        board.append(List)
    Put(n,board, 0)
def isSafe(n,board,row, col):
    for i in range(n):
        if board[row][i] == 1:
            return False
    for j in range(n):
        if board[j][col] == 1:
            return False
# check left up diagonal
    i = row-1
    j = col-1
    while(i >= 0 and j >= 0):
        if board[i][j] == 1:
            return False
        i = i-1
        j = j-1
# check right up diagonal

    i = row-1
    j = col+1
    while(i >= 0 and j < n):
        if board[i][j] == 1:
            return False
        i = i-1
        j = j+1

    i = row+1
    j = col-1
    while(i < n and j >= 0):
        if board[i][j] == 1:
            return False
        i = i+1
        j = j-1

    i = row+1
    j = col+1
    while(i < n and j < n):
        if board[i][j] == 1:
            return False
        i = i+1
        j = j+1
    return True
def Put(n,board, count):
    if count == n:
        Geneticsolution(n,board)
        return

    for i in range(len(board)):
        if isSafe(n,board,count, i):
            board[count][i] = 1
            Put(n,board, count + 1)
            board[count][i] = 0

def StartBackTrackinginThread(BackBoard,n1,BackboardQueue,MAxBQueue,SolutionBackQ,IMAGE,EQ):
    solutionsBACK.clear()
    print(backtracking(n1))
    BackBoard = CreateBoard(n1, 400, solutionsBACK[0], IMAGE)
    MAxBQueue.put(len(solutionsBACK))
    SolutionBackQ.put(solutionsBACK)
    EQ.put(time.thread_time())
    BackboardQueue.put(BackBoard)
def Geneticsolution(n,board):
    solution=[]
    for i in range(0,n) :
        counter = 0
        for j in board:
            if j[i]==1 :
                solution.append(counter)
                break
            else:
                counter+=1
    solutionsBACK.append(solution)

def CreateBoard(n,size,Solution,IMAGE):
    IMAGE = pygame.transform.rotozoom(IMAGE, 0, 0.4/n)
    board = pygame.Surface((size, size))
    board.fill(pygame.Color(255, 0, 0))
    cellsize = int(size / n)
    color = 1
    imagex,imagey=IMAGE.get_size()
    for i in range(0, n * cellsize, cellsize):
        for j in range(0, n * cellsize, cellsize):
            if color == 1:
                if int(j/ cellsize) in Solution and Solution.index(int(j/cellsize)) == int(i/cellsize) :
                    pygame.draw.rect(board, pygame.Color(159, 69, 34), pygame.Rect((i, j, cellsize, cellsize)))
                    board.blit(IMAGE,(((i+((cellsize-imagex)/2)),(j+((cellsize-imagey)/2)))))
                else:
                    pygame.draw.rect(board, pygame.Color(159, 69, 34), pygame.Rect((i, j, cellsize, cellsize)))
                color = -1
            else:
                if int(j/ cellsize) in Solution and Solution.index(int(j/cellsize)) == int(i/cellsize) :
                    pygame.draw.rect(board, pygame.Color(232, 210, 159), pygame.Rect((i, j, cellsize, cellsize)))
                    board.blit(IMAGE,((i+((cellsize-imagex)/2)),(j+((cellsize-imagey)/2))))
                else:
                    pygame.draw.rect(board, pygame.Color(232, 210, 159), pygame.Rect((i, j, cellsize, cellsize)))
                color = 1
        if n % 2 == 0:
            color *= -1
    return board

def Gui(WindowSizeX,WindowSizeY):
    pygame.init()
    MainFont = pygame.font.SysFont("Consolas",20,False,True)
    pygame.display.set_caption('NQueen')
    window_surface = pygame.display.set_mode((WindowSizeX, WindowSizeY))

    background = pygame.Surface((WindowSizeX, WindowSizeY))
    background.fill('#353535')

    manager = pygame_gui.UIManager((WindowSizeX, WindowSizeY),'theme.json')

    ##### Start Button

    StartButtom = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(WindowSizeX/2-int((150/2)),700,150,50),
                                                text='Start Algorithm',
                                                manager=manager)
    #### NEXT BUTTON
    NextButtom = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(330, 600, 150, 50),
                                               text='NEXT',
                                               manager=manager)
    #### Previous BUTTON
    PreviousButtom = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(80, 600, 150, 50),
                                              text='PREVIOUS',
                                              manager=manager)

    #### NEXT BUTTON for Backtracking
    NextButtomB = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(1170, 600, 150, 50),
                                              text='NEXT',
                                              manager=manager)
    PreviousButtomB = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(920, 600, 150, 50),
                                                  text='PREVIOUS',
                                                  manager=manager)

    #### NEXT BUTTON
    ScatterButton = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(150, 655, 250, 35),
                                              text='Generations Compare',
                                              manager=manager)

    ShowAllComparisons = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(WindowSizeX/2-(250/2), 600, 250, 55),
                                                 text='Show Algorithms Comparison',
                                                 manager=manager)



    ##### INPUTS
    n_input = UITextEntryLine(relative_rect=pygame.Rect(WindowSizeX/4 - 150, 50, 300, 100), manager=manager)
    n_input.set_text("EnterN")
    gen_input = UITextEntryLine(relative_rect=pygame.Rect(WindowSizeX/2 - 150, 50, 300, 100), manager=manager)
    gen_input.set_text("Enter Generation Number")
    pop_input = UITextEntryLine(relative_rect=pygame.Rect((WindowSizeX*3)/4 - 150 , 50, 300, 100), manager=manager)
    pop_input.set_text("Enter Population Number")

    #Board
    GeneticBoard = pygame.Surface((0, 0))
    BackBoard = pygame.Surface((0, 0))

    # Text
    GeneticText = MainFont.render("Differential Evolution",1,(255,255,255))
    BackText = MainFont.render("BackTracking", 1, (255, 255, 255))

    GeneticSpeed = MainFont.render("Esitmated Time : Null",1,(255,255,255))
    BackTrackSPeed = MainFont.render("Esitmated Time : Null",1,(255,255,255))

    GeneticSolutions = MainFont.render("Solution Number : Null", 1, (255, 255, 255))
    BackTrackSolutions= MainFont.render("Solution Number : Null", 1, (255, 255, 255))

    GenSIndex = MainFont.render("Solution Number : Null",1,(255,255,255))
    BackSIndex = MainFont.render("Solution Number : Null",1,(255,255,255))

    IMAGE = pygame.image.load("White.png").convert_alpha()

    ITERATIONS=0
    MAX=0

    ITERATIONSB = 0
    MAXB = 0


    #Queues
    MAXBQueue = queue.Queue() # Queue to receive length of backtracking solution from t2
    MAXQueue = queue.Queue() # Queue to receive length of differential solution from t1
    BackBoardQueue = queue.Queue()
    SolutionBQueue = queue.Queue()
    GenBoardQueue = queue.Queue()
    SolutionQueue = queue.Queue()
    EndTime1Q = queue.Queue()
    EndTime2Q = queue.Queue()

    OgGraded = queue.Queue()
    ResGraded = queue.Queue()

    AllDIFSolutionL = []
    AllBACKSolutionL = []

    is_running = True
    clock = pygame.time.Clock()
    while is_running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == StartButtom:
                        NoErrors = True
                        try:
                            n_input.text_colour = (255,255,255)
                            int(n_input.get_text())
                            n_input.text_colour = (0, 255, 0)
                        except ValueError:
                            n_input.text_colour = (255,0,0)
                            NoErrors = False
                        try:
                            gen_input.text_colour = (255,255,255)
                            int(gen_input.get_text())
                            gen_input.text_colour = (0, 255, 0)
                        except ValueError:
                            gen_input.text_colour = (255,0,0)
                            NoErrors = False
                        try:
                            pop_input.text_colour = (255,255,255)
                            int(pop_input.get_text())
                            pop_input.text_colour = (0, 255, 0)
                            if (int(pop_input.get_text()) < 4):
                                NoErrors = False
                                pop_input.text_colour = (255, 0, 0)
                        except ValueError:
                            pop_input.text_colour = (255,0,0)
                            NoErrors = False
                            pop_input.text_colour = (255, 0, 0)
                        if NoErrors:
                            ########################################## RESET EVERYTHING FIRST ###########################################
                            OgGraded = queue.Queue()
                            ResGraded = queue.Queue()
                            GenBoardQueue = queue.Queue()
                            MAXQueue = queue.Queue()
                            MAXBQueue = queue.Queue()
                            SolutionQueue = queue.Queue()
                            SolutionBQueue = queue.Queue()
                            EndTime1Q = queue.Queue()
                            EndTime2Q = queue.Queue()
                            GeneticBoard = None
                            BackBoard = None
                            ITERATIONS = 0
                            ITERATIONSB = 0
                            MAX = 0
                            MAXB = 0
                            solutionsBACK = []
                            ########################################################################

                            t1 = threading.Thread(target=StartAlgoinThread, args=(GeneticBoard, int(n_input.get_text()), int(pop_input.get_text()), int(gen_input.get_text()),IMAGE, GenBoardQueue,MAXQueue,SolutionQueue,EndTime1Q,OgGraded,ResGraded,AllDIFSolutionL))
                            t1.start()

                            t2 = threading.Thread(target=StartBackTrackinginThread,args=(BackBoard,int(n_input.get_text()),BackBoardQueue,MAXBQueue,SolutionBQueue,IMAGE,EndTime2Q))
                            t2.start()

                            NextButtom.disable()
                            NextButtomB.disable()
                            PreviousButtom.disable()
                            PreviousButtomB.disable()
                            ShowAllComparisons.disable()

                    if event.ui_element == NextButtom:
                        if MAXQueue.empty() == False:
                            MAX = MAXQueue.get()
                        if SolutionQueue.empty() == False:
                            Solutions = SolutionQueue.get()
                        if (ITERATIONS + 1 < MAX):
                            ITERATIONS += 1
                            GeneticBoard = None
                            GeneticBoard = CreateBoard(int(n_input.get_text()), 400, Solutions[ITERATIONS], IMAGE)
                            GenSIndex = MainFont.render("Solution Number : " + str(ITERATIONS), 1,(255, 255, 255))
                    if event.ui_element == PreviousButtom:
                        if MAXQueue.empty() == False:
                            MAX = MAXQueue.get()
                        if SolutionQueue.empty() == False:
                            Solutions = SolutionQueue.get()
                        if (ITERATIONS - 1 >= 0):
                            ITERATIONS -= 1
                            GeneticBoard = None
                            GeneticBoard = CreateBoard(int(n_input.get_text()), 400, Solutions[ITERATIONS], IMAGE)
                            GenSIndex = MainFont.render("Solution Number : " + str(ITERATIONS), 1,(255, 255, 255))

                    if event.ui_element == NextButtomB:
                        if MAXBQueue.empty() == False:
                            MAXB = MAXBQueue.get()
                        if SolutionBQueue.empty() == False:
                            solutionsBACK = SolutionBQueue.get()
                        if (ITERATIONSB + 1 < MAXB):
                            ITERATIONSB += 1
                            BackBoard = None
                            BackSIndex = MainFont.render("Solution Number : " + str(ITERATIONSB), 1,
                                                         (255, 255, 255))
                            BackBoard = CreateBoard(int(n_input.get_text()), 400, solutionsBACK[ITERATIONSB], IMAGE)
                    if event.ui_element == PreviousButtomB:
                        if MAXBQueue.empty() == False:
                            MAXB = MAXBQueue.get()
                        if SolutionBQueue.empty() == False:
                            solutionsBACK = SolutionBQueue.get()
                        if (ITERATIONSB - 1 >= 0):
                            ITERATIONSB -= 1
                            BackBoard = None
                            BackSIndex = MainFont.render("Solution Number : " + str(ITERATIONSB), 1,(255, 255, 255))
                            BackBoard = CreateBoard(int(n_input.get_text()), 400, solutionsBACK[ITERATIONSB], IMAGE)

                    if event.ui_element == ScatterButton and OgGraded.empty() != True and ResGraded.empty() != True:
                        ## take Original Array and Original Fitness
                        ogArray = np.array(OgGraded.get())
                        ogFitness = np.array(OgGraded.get())
                        ## take New Array with new Fitness
                        newarr = ResGraded.get()

                        # return everything back in case of Clicking the button again
                        OgGraded.put(ogArray)
                        OgGraded.put(ogFitness)
                        ResGraded.put(newarr)

                        UniqueSol = []
                        for x in range(0,len(ogArray)):
                            plt.scatter(ogFitness[x],x ,c="Red",zorder = 2) # Solution number(x) on y-axis
                            if (newarr[x] not in UniqueSol and newarr[x][0] == 0):
                                plt.scatter(newarr[x][0],x , c="LightGreen",zorder = 2)
                                UniqueSol.append(newarr[x])
                            else:
                                plt.scatter(newarr[x][0],x , c="Green",zorder = 2)
                            if newarr[x][0] != ogFitness[x]:
                                plt.plot([newarr[x][0],ogFitness[x]],[x,x],c="Blue",zorder = 1)
                        red_patch = mpatches.Patch(color='red', label='Initial Fitness')
                        green_patch = mpatches.Patch(color='Green',label= "Solution")
                        lightG_patch = mpatches.Patch(color='LightGreen',label= "Unique Solution")
                        plt.legend(handles=[red_patch,green_patch,lightG_patch])
                        plt.xlabel("Solution")
                        plt.ylabel("Fitness")
                        plt.show()

                    if event.ui_element == ShowAllComparisons:
                        ## Show Comparisons between Alldifsol and Allbacksol
                        for x in range(0,len(AllDIFSolutionL)):

                            plt.scatter(AllDIFSolutionL[x][0],AllDIFSolutionL[x][1],c="Purple")
                            if x != 0:
                                plt.plot([AllDIFSolutionL[x][0],AllDIFSolutionL[x-1][0]],[AllDIFSolutionL[x][1],AllDIFSolutionL[x-1][1]],c="Purple")

                        for y in range(0,len(AllBACKSolutionL)):
                            plt.scatter(AllBACKSolutionL[y][0],AllBACKSolutionL[y][1],c="Black")
                            if y!=0:
                                plt.plot([AllBACKSolutionL[y][0],AllBACKSolutionL[y-1][0]],[AllBACKSolutionL[y][1],AllBACKSolutionL[y-1][1]],c="Black")
                        purple_patch = mpatches.Patch(color='Purple', label='Differential')
                        black_patch = mpatches.Patch(color='Black', label="Backtracking")
                        plt.legend(handles=[purple_patch, black_patch])
                        plt.xlabel("Number of Solutions")
                        plt.ylabel("Time")
                        plt.show()


            manager.process_events(event)
        manager.update(time_delta)
        # Position Surfaces
        window_surface.blit(background, (0, 0))
        if GenBoardQueue.empty() == False:
            GeneticBoard = GenBoardQueue.get()
            GeneticSpeed = MainFont.render("TIME : " + str(EndTime1Q.get())+ " Second",1,(255,255,255))
            m = MAXQueue.get()
            GeneticSolutions = MainFont.render("No. of Solutions : "+str(m), 1, (255, 255, 255))
            GenSIndex = MainFont.render("Solution Number : " + str(ITERATIONS),1,(255,255,255))
            MAXQueue.put(m)
            NextButtom.enable()
            PreviousButtom.enable()
            if t1.is_alive() == False and t2.is_alive() == False:
                ShowAllComparisons.enable()

        if GeneticBoard != None:
            window_surface.blit(GeneticBoard, (80, 190))

        if BackBoardQueue.empty() == False: # BackTracking Found Solution
            EndTime = EndTime2Q.get()
            NumberofSolutions = MAXBQueue.get()
            AllBACKSolutionL.append((NumberofSolutions,EndTime))
            BackBoard = BackBoardQueue.get()
            BackTrackSPeed = MainFont.render("TIME : " + str(EndTime) + " Second",1,(255,255,255))
            BackTrackSolutions = MainFont.render("No. of Solutions : "+str(NumberofSolutions), 1, (255, 255, 255))
            BackSIndex = MainFont.render("Solution Number : " + str(ITERATIONSB), 1, (255, 255, 255))
            MAXBQueue.put(NumberofSolutions)
            NextButtomB.enable()
            PreviousButtomB.enable()
            if t1.is_alive() == False and t2.is_alive() == False:
                ShowAllComparisons.enable()
        if BackBoard != None:
            window_surface.blit(BackBoard, (920, 190))

        x,y=BackText.get_size()
        size=(400-x)/2

        #Main Text
        window_surface.blit(BackText, (920 + size, 120))
        window_surface.blit(GeneticText, (160, 120))

        #Solution Index
        window_surface.blit(GenSIndex,(160,155))
        window_surface.blit(BackSIndex,(920 + (size/2),155))

        #Speed Text
        window_surface.blit(GeneticSpeed, (80, 700))
        window_surface.blit(BackTrackSPeed, (920, 700))

        #Solutions All Number
        window_surface.blit(GeneticSolutions, (80, 730))
        window_surface.blit(BackTrackSolutions, (920, 730))

        # Draw UI
        manager.draw_ui(window_surface)
        #Update ALL

        pygame.display.update()
Gui(1400,800)
