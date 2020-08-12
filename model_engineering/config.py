########## Configs ##########

###### Imports ######

###### Constants ###### 
CURRENT_OS = 'Windows'                # manually change this field every time before deploying (maybe at gdrive)
workspace = "/model_engineering"

if CURRENT_OS.lower()[0] == 'w':
  DIR = '.'
  DIR_OUTPUT = '.\\outputs'
elif CURRENT_OS.lower()[0] == 'c':
  DIR = 'drive/My Drive/Live Workspace/generative-facial-cosmetics/model_engineering/'
  print("Output directories will be moved to GCS instead; make sure to authenticate GCS")
  DIR_OUTPUT = "gs://" + "generative-facial-cosmetics" + workspace + "/"
else:
  raise NotImplementedError('OS is not supported yet')

###### Functions ######
def isWindows():
  return bool(CURRENT_OS.lower()[0] == 'w')
def isColab():
  return bool(CURRENT_OS.lower()[0] == 'c')

###### Execution ######
