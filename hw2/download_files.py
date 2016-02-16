#!/usr/bin/env python

import os

# all of our read files
READ_URL_BASE='ftp://ftp.sra.ebi.ac.uk/vol1/fastq/ERR192/'
READ_URL_NUMS=( 'ERR192339',
                'ERR192340',
                'ERR192341',
                'ERR192342',
                'ERR192343',
                'ERR192344',
                'ERR192345',
                'ERR192346',
                'ERR192347',
                'ERR192348' )

# all of our chromosome files
CHRM_URL_BASE='http://hgdownload.cse.ucsc.edu/goldenPath/mm10/chromosomes/'
CHRM_URL_NUMS=( 'chr1',
                'chr2',
                'chr3',
                'chr4',
                'chr5',
                'chr6',
                'chr7',
                'chr8',
                'chr9',
                'chr10',
                'chr11',
                'chr12',
                'chr13',
                'chr14',
                'chr15',
                'chr16',
                'chr17',
                'chr18',
                'chr19' )

def download(url, dir=''):
    '''This function is heavily influenced from the following:
    http://stackoverflow.com/questions/22676/how-do-i-download-a-file-over-http-using-python'''

    import urllib2

    # determine the filename
    filename = url.split('/')[-1]
    if (dir):
        filename = os.path.join(dir, filename)

    # open the url and get the meta info
    u = urllib2.urlopen(url)
    meta = u.info()
    filesize = int(meta.getheaders("Content-Length")[0])

    # see if we've already downloaded
    if os.path.exists(filename) or os.path.exists(filename[:-3]):
        print "File already downloaded. Skipping %s" % (filename)
        return True

    # open the file for writing
    f = open(filename, 'wb')
    print "Downloading: %s | Bytes: %s" % (filename, filesize)

    # status bar for the download
    filesize_dl = 0
    blocksize = 8192
    while True:
        # download the next chunk
        buffer = u.read(blocksize)
        if not buffer:
            break

        # update our download size
        filesize_dl += len(buffer)
        f.write(buffer)

        # print out a status bar
        status = r"%10d  [%3.2f%%]" % (filesize_dl, filesize_dl * 100. / filesize)
        status = status + chr(8) * (len(status)+1)
        print status,

    # close the file and return success status
    f.close()
    return filesize == filesize_dl

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download genome files for UND CS532 homework 2')
    parser.add_argument('--count', type=int, default='0', help='max number of files to download (<=0 for all)')

    args = parser.parse_args()
    if args.count <= 0:
        args.count = 0

    # create our data directory
    dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(dir, 'data')
    if not os.path.exists(dir):
        os.makedirs(dir)

    # download all the read files
    i = 0
    for num in READ_URL_NUMS:
        if args.count and i == args.count:
            break
        download("%s/%s/%s.fastq.gz" % (READ_URL_BASE, num, num), dir)
        i += 1

    # download all the chromosome files
    i = 0
    for num in CHRM_URL_NUMS:
        if args.count and i == args.count:
            break
        download("%s/%s.fa.gz" % (CHRM_URL_BASE, num), dir)
        i += 1
