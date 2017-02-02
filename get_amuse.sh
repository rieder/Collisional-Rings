#! /bin/bash

# Download and extract AMUSE
wget https://github.com/amusecode/amuse/archive/036fe8562c63145d4546dc9f9eefe948b4a89f67.zip && \
unzip 036fe8562c63145d4546dc9f9eefe948b4a89f67.zip && \
mv amuse-036fe8562c63145d4546dc9f9eefe948b4a89f67 amuse && \
rm 036fe8562c63145d4546dc9f9eefe948b4a89f67.zip 

# Patch rebound makefile to compile on my Macbook
cd amuse/src/amuse/community/rebound/
python download_http.py 
cd ../../../../../
cp patches/rebound_Makefile.defs amuse/src/amuse/community/rebound/src/rebound/src/Makefile.defs 
cp patches/rebound_interface.cc amuse/src/amuse/community/rebound/interface.cc

# Build AMUSE
cd amuse/
./configure && \
make framework && \
make rebound.code

cd ..
