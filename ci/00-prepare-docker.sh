#!/bin/sh

# intel repos are frequently out of sync, to a point that makes them        
# barely usable. And anyway we don't care: there's no software that         
# we want to pull from these repos anyway.                                  
find /etc/apt/sources.list.d/ -type f | xargs -r grep -li intel | xargs -r rm
apt-get update -qq && apt-get install -qq build-essential git libmpfr-dev
# the default MPFR library does not contain new C23 functions
# we thus install MPFR 4.2.1
wget https://www.mpfr.org/mpfr-current/mpfr-4.2.2.tar.gz
tar xf mpfr-4.2.2.tar.gz
cd mpfr-4.2.2
./configure
make -j
make install
