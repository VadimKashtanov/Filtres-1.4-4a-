#! /usr/bin/python3

from mdl import *

if __name__ == "__main__":
	mdl = Mdl("mdl_C17_perf98_3.bin")

	for i in range(5):
		mdl.incruster_DOT1D(7, 64)

	for i in range(5):
		mdl.incruster_DOT1D(8, 32)

	mdl.ecrire("mdl.bin")