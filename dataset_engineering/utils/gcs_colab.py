##### GCS Integration module (for windows) #####

#### imports

#### constants

#### functions
def init():
  ''' under construction
  ## install gcs fuse
  !echo "deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  !curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  !sudo apt-get -y -q update
  !sudo apt-get -y -q install gcsfuse
  ## authentication
  from google.colab import auth
  auth.authenticate_user()
  '''

## test pinging to each regions
def ping_regions():
  '''
  !curl https://storage.googleapis.com/gcping-release/gcping_linux_amd64_0.0.3 > gcping && chmod +x gcping
  !./gcping
  '''
## mount gcs bucket
def mount_bucket(bucketname):
  '''
  storepath = "./gcs_mounts/" + bucketname
  !mkdir -p gcs_mounts
  !mkdir -p {storepath}
  !gcsfuse --implicit-dirs --limit-bytes-per-sec -1 --limit-ops-per-sec -1 {bucketname} {storepath}
  '''
## unmount gcs bucket
def unmount_bucket(bucketname):
  '''
  storepath = "./gcs_mounts/" + bucketname
  !umount storepath
  '''