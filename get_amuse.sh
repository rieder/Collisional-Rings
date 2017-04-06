#! /bin/bash

# Download and extract AMUSEa
export GITHASH=e2ab1eeb919b29ff8c9686da1c8d925f9e7a3928
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
