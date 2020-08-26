##### GCS Integration module (for windows) #####

#### imports

#### constants

workspace = "\\model_engineering"

#### functions
def init():
  return

## test pinging to each regions
def ping_regions():
  raise NotImplementedError('Sorry :(')

## mount gcs bucket
def mount_bucket(bucketname, selectedVol=None):
  if selectedVol != None:
    return selectedVol + ":"
  print('Enter this command to your terminal and type in the volume of the mount:')
  print('rclone mount remote:' + bucketname + ' <VOL>: --vfs-cache-mode writes')
  vol = input('Mounted volume >> ')
  return vol + ':\\', vol + ':\\'             # special route placeholder, normal route

## unmount gcs bucket
def unmount_bucket(bucketname):
  print('Abort the server of that bucket in your terminal')
  
#### execution