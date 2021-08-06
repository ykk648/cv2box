#!/bin/bash

export RSH_RSYNC_USER_151=
export RSH_RSYNC_USER_PASSWORD_151=
export REMOTE_SERVER_151=
export SYNC_151_DIR=/nfs/cv_rsync/

export RSH_RSYNC_USER_84=
export RSH_RSYNC_USER_PASSWORD_84=
export REMOTE_SERVER_84=
export SYNC_84_DIR=/mnt/cv_rsync/


export RSH_RSYNC_USER_85=
export RSH_RSYNC_USER_PASSWORD_85=
export REMOTE_SERVER_85=
export SYNC_85_DIR=/mnt/cv_rsync/


export RSH_RSYNC_PORT=22


if [ $1x = "84"x ]; then
  sshpass -p $RSH_RSYNC_USER_PASSWORD_151 rsync -avzP --delete -e "ssh -p $RSH_RSYNC_PORT" $RSH_RSYNC_USER_151@$REMOTE_SERVER_151:$SYNC_151_DIR $SYNC_84_DIR
elif [ $1x = "85"x ]; then
  sshpass -p $RSH_RSYNC_USER_PASSWORD_151 rsync -avzP --delete -e "ssh -p $RSH_RSYNC_PORT" $RSH_RSYNC_USER_151@$REMOTE_SERVER_151:$SYNC_151_DIR $SYNC_85_DIR
elif [ $1x = "151"x ]; then
  sshpass -p $RSH_RSYNC_USER_PASSWORD_84 rsync -avzP -e "ssh -p $RSH_RSYNC_PORT" $RSH_RSYNC_USER_84@$REMOTE_SERVER_84:$SYNC_84_DIR $SYNC_151_DIR
else
  echo plz add param: ['84', '151', '85']!
fi