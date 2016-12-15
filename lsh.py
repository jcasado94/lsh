#!/usr/bin/env python

"""
Simple module implementing LSH
"""

__version__ = '0.2'
__author__  = 'marias@cs.upc.edu'

import numpy
import sys
import argparse
import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print ('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result

    return timed


class lsh(object):
    """
    implements lsh for digits database in file 'images.npy'
    """  
    
    def __init__(self, k, m):
        """ k is nr. of bits to hash and m is reapeats """
        # data is numpy ndarray with images
        self.data = numpy.load('images.npy')
        self.k = k
        self.m = m

        # determine length of bit representation of images
        # use conversion from natural numbers to unary code for each pixel,
        # so length of each image is imlen = pixels * maxval
        self.pixels = 64
        self.maxval = 16
        self.imlen = self.pixels * self.maxval

        # need to select k random hash functions for each repeat
        # will place these into an m x k numpy array
        numpy.random.seed(12345)
        self.hashbits = numpy.random.randint(self.imlen, size=(m, k))

        # the following stores the hashed images
        # in a python list of m dictionaries (one for each repeat)
        self.hashes = [dict()] * self.m

        # now, fill it out
        self.hash_all_images()

        return
    

    def hash_all_images(self):
        """ go through all images and store them in hash table(s) """
        # Achtung!
        # Only hashing the first 1500 images, the rest are used for testing
        for idx, im in enumerate(self.data[:1500]):
            for i in range(self.m):
                str = self.hashcode(im, i)

                # store it into the dictionary.. 
                # (well, the index not the whole array!)
                if str not in self.hashes[i]:
                    self.hashes[i][str] = []
                self.hashes[i][str].append(idx)
        return


    def hashcode(self, im, i):
        """ get the i'th hash code of image im (0 <= i < m)"""
        pixels = im.flatten()
        row = self.hashbits[i]
        str = ""
        for x in row:
            # get bit corresponding to x from image..
            pix = int(x) / int(self.maxval)
            num = x % self.maxval
            if (num <= int(pixels[int(pix)])):
                str += '1'
            else:
                str += '0'
        return str


    def candidates(self, im):
        """ given image im, return matching candidates (well, the indices) """
        res = set()
        for i in range(self.m):
            code = self.hashcode(im, i)
            if code in self.hashes[i]:
                res.update(self.hashes[i][code])
        return res


@timeit
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default=20, type=int)
    parser.add_argument('-m', default=5, type=int)
    args = parser.parse_args()    

    print ("Running lsh.py with parameters k =", args.k, "and m =", args.m)

    me = lsh(args.k, args.m)

    # # show candidate neighbors for first 10 test images
    # for r in range(1500,1510):
    #     im = me.data[r]
    #     cands = me.candidates(im)
    #     print ("there are %4d candidates for image %4d" % (len(cands), r))


    print(getClosestImgBruteForce(me.data, 7))
    print(getClosestImgHashing(me, 7)) # different results, due to different methods

    print(getClosestImgBruteForce(me.data, 0))
    print(getClosestImgHashing(me, 0)) # same results

    return


def imgDif(im1, im2) :

	pixelsIm1 = im1.flatten()
	pixelsIm2 = im2.flatten()
	difference = 0

	for pixel in range(0, len(pixelsIm1)) :

		pixelIm1 = pixelsIm1[pixel]
		pixelIm2 = pixelsIm2[pixel]
		# 2 -> 110000000000000, 4 -> 1111000000000000 ----> difference = abs(pixelIm1 - pixelIm2)
		difference += abs(pixelIm1 - pixelIm2)

	return difference


def getClosestImgBruteForce(data, imIndex) :

    im = data[imIndex]
    closestIm = -1
    closestDiff = -1

    for ind in range(0, 1500):

        if ind == imIndex: 
            continue

        diff = imgDif(data[ind], im)
        if diff < closestDiff or closestDiff == -1 :
            closestIm = ind
            closestDiff = diff

    return (closestIm, closestDiff)


def getClosestImgHashing(me, imIndex):

    im = me.data[imIndex]
    closestIm = -1
    closestDiff = -1
    cands = me.candidates(im)
    if (len(cands) == 1): return None # itself

    for cand in cands:

        if (cand == imIndex): 
            continue

        diff = imgDif(me.data[cand], im)
        if diff < closestDiff or closestDiff == -1 :
            closestIm = cand
            closestDiff = diff

    return (closestIm, closestDiff)


if __name__ == "__main__":
  sys.exit(main())
