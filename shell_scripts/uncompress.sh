#sh uncompress.sh 'tar' ./ '000000'

if [ -n "$2" ]; then
  cd $2
fi

if [ $1 = "7z" ]; then
  for i in *.7z; do
    if [ -n "$3" ]; then
      7z x "$i" -p$3 -aos
    else
      7z x "$i"
    fi
  done
elif [ $1 = "tar" ]; then
  for i in *.tar; do
    tar -xvf "$i"
  done
elif [ $1 = "tar.gz" ]; then
  for i in *.tar.gz; do
    tar -zxvf "$i"
  done
elif [ $1 = "zip" ]; then
  for i in *.zip; do
    unzip -P $3 "$i"
  done
elif [ $1 = "7zip" ]; then
  for i in *.zip; do
    if [ -n "$3" ]; then
      7z x "$i" -p$3 -ao s
    else
      7z x "$i"
    fi
  done
fi