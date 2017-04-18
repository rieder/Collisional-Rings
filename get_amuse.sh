#! /bin/bash

# Download and extract AMUSEa
export GITHASH=e1f50d55b9fa102eb6aa028a6d75062bffa5d196
wget https://github.com/rieder/amuse/archive/${GITHASH}.zip && \
  unzip ${GITHASH}.zip && \
  mv amuse-${GITHASH} amuse && \
  rm ${GITHASH}.zip 

export SCRIPT_DIR="${PWD}"
export AMUSE_DIR="${PWD}/amuse"
export REBOUND_DIR="${AMUSE_DIR}/src/amuse/community/rebound"

# Patch rebound makefile to compile on my Macbook
cd ${REBOUND_DIR}
python download_http.py 

# Build AMUSE
cd ${AMUSE_DIR}
./configure && \
make framework && \
make rebound.code

cd ${SCRIPT_DIR}
