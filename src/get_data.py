#!/usr/bin/env python

import urllib2
import gzip
import os

def pull_file(url):
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()

def main():
    urls = ["http://personal.disco.unimib.it/Vanneschi/bioavailability.txt",
            "http://personal.disco.unimib.it/Vanneschi/bioavailability_lookup.txt",
            "http://personal.disco.unimib.it/Vanneschi/toxicity.txt",
            "http://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/tsp/ALL_tsp.tar.gz",
            "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html"
            # TODO download all the Hamiltonian cycle probems,
            # capacitated arc-routing problems, etc from the same
            # source as TSPLIB
            ]

    for url in urls:
        pull_file(url)
    os.makedirs("TSPLIB")
    os.rename("ALL_tsp.tar.gz", "TSPLIB/ALL_tsp.tar.gz")
    # TODO this should be platform-independent
    os.system("cd TSPLIB; tar xzf ALL_tsp.tar.gz")
    os.rename("STSP.html", "TSPLIB/STSP.html")
    
if __name__ == '__main__':
    os.makedirs("../data")
    os.chdir("../data")
    main()
