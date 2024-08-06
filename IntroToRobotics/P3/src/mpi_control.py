from megapi import MegaPi


MFR = 2     # port for motor front right
MBL = 3     # port for motor back left
MBR = 10    # port for motor back right
MFL = 11    # port for motor front left

class MegaPiController:
    def __init__(self, port='/dev/ttyUSB0', verbose=True):
        self.port = port
        self.verbose = verbose
        if verbose:
            self.printConfiguration()
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfr = MFR  # port for motor front right
        self.mbl = MBL  # port for motor back left
        self.mbr = MBR  # port for motor back right
        self.mfl = MFL  # port for motor front left   

    
    def printConfiguration(self):
        print('MegaPiController:')
        print("Communication Port:" + repr(self.port))
        print("Motor ports: MFR: " + repr(MFR) +
              " MBL: " + repr(MBL) + 
              " MBR: " + repr(MBR) + 
              " MFL: " + repr(MFL))

    # fah min 40 
    
    def setFourMotors(self, vfl=0, vfr=0, vbl=0, vbr=0, min_val=45):
        vfr = 1 * int(vfr)
        vbr = 1 * int(vbr)

        vfr = self.setMin(vfr, min_val=min_val)
        vbr = self.setMin(vbr, min_val=min_val)
        vfl = self.setMin(vfl, min_val=min_val)
        vbl = self.setMin(vbl, min_val=min_val)

        vfl = int(vfl * 1.1)
        vbl = int(vbl * 1.1)

        vfl = -1 * vfl
        vbl = -1 * vbl
    
        if self.verbose:
            print("Set Motors: vfl: " + repr(int(round(vfl,0))) + 
                  " vfr: " + repr(int(round(vfr,0))) +
                  " vbl: " + repr(int(round(vbl,0))) +
                  " vbr: " + repr(int(round(vbr,0))))
        self.bot.motorRun(self.mfl,vfl)
        self.bot.motorRun(self.mfr,vfr)
        self.bot.motorRun(self.mbl,vbl)
        self.bot.motorRun(self.mbr,vbr)

    def setMin(self, item, min_val):
        if abs(item) > min_val:
            return item

        if abs(item) <= 0.1:
            return 0

        if item < 0:
            return -min_val
        
        if item > 0:
            return min_val
        

    def setFourMotorsCalibrated(self, vfl=0, vfr=0, vbl=0, vbr=0, min_val=45, disable=False):
        vfr = 1 * int(vfr)
        vbr = 1 * int(vbr)

        vfr = self.setMin(vfr, min_val=min_val)
        vbr = self.setMin(vbr, min_val=min_val)
        vfl = self.setMin(vfl, min_val=min_val)
        vbl = self.setMin(vbl, min_val=min_val)

        vfl = int(vfl * 1.1)
        vbl = int(vbl * 1.1)

        vfl = -1 * vfl
        vbl = -1 * vbl

        if self.verbose:
            print("Set Motors Calibrated: vfl: " + repr(int(round(vfl,0))) + 
                  " vfr: " + repr(int(round(vfr,0))) +
                  " vbl: " + repr(int(round(vbl,0))) +
                  " vbr: " + repr(int(round(vbr,0))))

        if not disable:
            self.bot.motorRun(self.mfl,vfl)
            self.bot.motorRun(self.mfr,vfr)
            self.bot.motorRun(self.mbl,vbl)
            self.bot.motorRun(self.mbr,vbr)    

    def carStop(self):
        if self.verbose:
            print("CAR STOP:")
        self.setFourMotors()


    def carStraight(self, speed):
        if self.verbose:
            print("CAR STRAIGHT:")
        self.setFourMotors(-speed, speed, -speed, speed)


    def carRotate(self, speed):
        if self.verbose:
            print("CAR ROTATE:")
        self.setFourMotors(speed, speed, speed, speed)


    def carSlide(self, speed):
        if self.verbose:
            print("CAR SLIDE:")
        self.setFourMotors(speed, speed, -speed, -speed)

    
    def carMixed(self, v_straight, v_rotate, v_slide):
        if self.verbose:
            print("CAR MIXED")
        self.setFourMotors(
            v_rotate-v_straight+v_slide,
            v_rotate+v_straight+v_slide,
            v_rotate-v_straight-v_slide,
            v_rotate+v_straight-v_slide
        )
    
    def close(self):
        self.bot.close()
        self.bot.exit()


if __name__ == "__main__":
    import time
    mpi_ctrl = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    time.sleep(1)
    # mpi_ctrl.carStraight(30)
    mpi_ctrl.setFourMotors(0,0,0,0)
    time.sleep(4)
    # mpi_ctrl.carSlide(30)
    # time.sleep(1)
    # mpi_ctrl.carRotate(30)
    # time.sleep(1)
    mpi_ctrl.carStop()
    # print("If your program cannot be closed properly, check updated instructions in google doc.")
