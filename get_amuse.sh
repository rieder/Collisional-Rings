#! /bin/bash

# Download and extract AMUSEa
export GITHASH=4dccd5460fc37f0663307250ac3073ac91b49331
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
cd ${SCRIPT_DIR}
cp patches/rebound_Makefile.defs ${REBOUND_DIR}/src/rebound/src/Makefile.defs 
cp patches/rebound_interface.cc ${REBOUND_DIR}/interface.cc

# Build AMUSE
cd ${AMUSE_DIR}
./configure && \
make framework && \
make rebound.code

cd ..
