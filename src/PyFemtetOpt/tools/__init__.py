# -*- coding: utf-8 -*-
from time import sleep
import os
here, me = os.path.split(__file__)

def activateFemtetPythonMacro():
    import subprocess
    from win32com.client import makepy
    library = 'FemtetMacro'
    cmd = ['python', makepy.__file__, library]
    subprocess.run(cmd)
    print('Spyder などの IDE を使っている場合、再起動してください。')


##class FemtetRefXla:
##    def __init__(self, visible=False):
##        from win32com.client import DispatchEx
##        # Excelのインスタンスを開始
##        ExcelApp = DispatchEx("Excel.Application")
##        ExcelApp.visible = visible
##        ExcelApp.DisplayAlerts = visible
##        self.wb = ExcelApp.Workbooks.Open(os.path.join(here, 'FemtetUtil.xlsm'))
##        self.app = ExcelApp
##    
##    def __del__(self):
##        self.app.Quit()
##    
##    def ExecFemtet(self):
##        ret = self.app.Run('ExecFemtet')
##        sleep(5)
##        return ret
##
##    def CloseFemtet(self):
##        return self.app.Run('CloseFemtet')
##        
##    def GetFemtetPath(self):
##        return self.app.Run('GetFemtetPath')
##
##    def GetFemtetActivate(self):
##        return self.app.Run('GetFemtetActivate')
##
##    def ExecProcess(self, command:str):
##        return self.app.Run('ExecProcess', command)
##
##    def CloseProcess(self):
##        return self.app.Run('CloseProcess')
##
##    def GetHProcess(self):
##        return self.app.Run('GetHProcess')
##    
##    def ExecProcessAndWait(self, command:str, exitCode:int):
##        return self.app.Run('ExecProcessAndWait', command, exitCode)
    



    
