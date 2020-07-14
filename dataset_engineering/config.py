########## Configs ##########

###### Imports ######

###### Constants ######
CURRENT_OS = 'Windows'

if CURRENT_OS.lower()[0] == 'w':
  DIR = '.'
elif CURRENT_OS.lower()[0] == 'c':
  DIR = 'drive/My Drive/Live Workspace/generative-facial-cosmetics/'
else:
  raise NotImplementedError('OS is not supported yet')

###### Functions ######
def isWindows():
  return bool(CURRENT_OS.lower()[0] == 'w')
def isColab():
  return bool(CURRENT_OS.lower()[0] == 'c')

###### Execution ######
