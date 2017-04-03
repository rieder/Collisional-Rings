#! /bin/bash

# Download and extract AMUSEa
export GITHASH=ba6280804d15248156e4eb923d5b0f6e1a710b31
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
