#! /bin/bash

# Download and extract AMUSE
wget https://github.com/amusecode/amuse/archive/036fe8562c63145d4546dc9f9eefe948b4a89f67.zip && \
unzip 036fe8562c63145d4546dc9f9eefe948b4a89f67.zip && \
mv amuse-036fe8562c63145d4546dc9f9eefe948b4a89f67 amuse && \
rm 036fe8562c63145d4546dc9f9eefe948b4a89f67.zip 

export SCRIPT_DIR="$(PWD)"
export AMUSE_DIR="$(PWD)/amuse"
export REBOUND_DIR="$(AMUSE_DIR)/src/amuse/community/rebound"

# Patch rebound makefile to compile on my Macbook
cd $(REBOUND_DIR)
python download_http.py 
cd $(SCRIPT_DIR)
cp patches/rebound_Makefile.defs $(REBOUND_DIR)/src/rebound/src/Makefile.defs 
cp patches/rebound_interface.cc $(REBOUND_DIR)/interface.cc

# Build AMUSE
cd $(AMUSE_DIR)
./configure && \
make framework && \
make rebound.code

cd ..
