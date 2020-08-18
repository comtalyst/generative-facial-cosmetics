##### GCS Integration module (for windows) #####

#### imports
import os

#### constants

workspace = "/encoder_engineering"

#### functions
def init():
  ## install gcs fuse
  os.system("echo deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main | sudo tee /etc/apt/sources.list.d/gcsfuse.list")
  os.system("curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -")
  os.system("sudo apt-get -y -q update")
  os.system("sudo apt-get -y -q install gcsfuse")
  ## authentication
  from google.colab import auth
  auth.authenticate_user()

## test pinging to each regions
def ping_regions():
  raise NotImplementedError()
  '''
  !curl https://storage.googleapis.com/gcping-release/gcping_linux_amd64_0.0.3 > gcping && chmod +x gcping
  !./gcping
  '''
## mount gcs bucket
def mount_bucket(bucketname):
  storepath = "./gcs_mounts/" + bucketname
  os.system("mkdir -p gcs_mounts")
  os.system("mkdir -p " + storepath + "")
  os.system("gcsfuse --implicit-dirs --limit-bytes-per-sec -1 --limit-ops-per-sec -1 " + bucketname + " " + storepath)
  return "gs://" + bucketname + workspace, storepath + workspace        # special route for colab tpu, normal route

## unmount gcs bucket
def unmount_bucket(bucketname):
  raise NotImplementedError()
  '''
  storepath = "./gcs_mounts/" + bucketname
  !umount storepath
  '''