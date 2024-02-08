import struct as st

from bitget import *

from os import system

import numba
from numba import njit, prange

from math import exp, tanh
import struct as st

from random import random, randint

rnd = lambda : 2*random()-1

def lire_uint(I, _bin):
	l = list(st.unpack('I'*I, _bin[:st.calcsize('I')*I]))
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = list(st.unpack('f'*I, _bin[:st.calcsize('f')*I]))
	return l, _bin[st.calcsize('f')*I:]

def norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [(e-_min)/(_max-_min) for e in arr]

def e_norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [2*(e-_min)/(_max-_min)-1 for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]:
		e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

with open("structure_generale.bin", "rb") as co:
	bins = co.read()

	constantes, bins = lire_uint(18, bins)

	exec("""P,
		P_INTERV,
		N,
		MAX_INTERVALLES,
		MAX_DECALES,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
		""".replace('\n', '') + " = constantes")

	min_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)
	max_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)

	NATURE_PARAMS, bins = lire_uint(NATURES, bins)

MIN_EMA = 1
MIN_NATURES = 0
MIN_NATURES = NATURES-1
MIN_INTERVALLES = 1

DEPART = (N+MAX_DECALES)*MAX_INTERVALLES

########################

def filtre_prixs__poids(Y,X):
	return F_PAR_BLOQUES*BLOQUES*N#Y*N

def dot1d__poids(Y,X):
	return (X+1)*Y

########################

inst_poids = [
	filtre_prixs__poids,
	dot1d__poids,
	0,
	0
]

########################

### Natures ###
#@njit(parallel=True)
def __nature__ema(source, params):
	return source, None

#@njit(parallel=True)
def __nature__macd(source, params):
	coef,_,_,_ = params
	#
	assert coef > 0.0
	ema12 = ema(source, 12*coef);
	ema26 = ema(source, 26*coef);
	_macd = [ema12[i]-ema26[i] for i in range(len(source))]
	ema9  = ema(_macd,  12*coef);
	return [_macd[i] - ema9[i] for i in range(len(source))], None

#@njit(parallel=True)
def __nature__chiffre(source, params):
	D,_,_,_ = params
	return [2*(D-min([abs(x-D*round((x+0)/D)), abs(x-D*round((x+D)/D))]))/D-1 for x in source], (0, D/2)

I__natures = [
	__nature__ema,
	__nature__macd,
	__nature__chiffre
]

########################

class Mdl:
	def __init__(self, fichier):
		with open(fichier, "rb") as co:
			bins = co.read()

		self.Y,     bins = lire_uint(C, bins)
		self.insts, bins = lire_uint(C, bins)

		for inst in self.insts:
			assert inst <= 1

		self.ema_int = []
		self.lignes  = []
		for _ in range(BLOQUES):
			(source, nature, K_ema, intervalle, decale), bins = lire_uint(5, bins)
			params, bins = lire_uint(MAX_PARAMS, bins)

			self.ema_int += [{
				'source'     : source,
				'nature'     : nature,
				'K_ema'      : K_ema,
				'intervalle' : intervalle,
				'decale'     : decale,
				'params'     : params
			}]
			
			assert nature <= max([0,1,2])

			self.lignes += [
				I__natures[nature]( ema(I__sources[source], K_ema), params)
			]
		
		self.p = []
		self.poids = []
		for i in range(C):
			X, Y = (self.Y[i-1] if i!=0 else 0), self.Y[i]
			self.poids += [inst_poids[self.insts[i]](Y,X)]
			poids, bins = lire_flotants(self.poids[i], bins)
			self.p += [poids]

	def incruster_DOT1D(self, apres, Y):
		global C
		print(f"Désormais C={len(self.Y)+1} au lieux de C={C}")
		C += 1
		#
		X = self.Y[apres]
		p = [(2*rnd()-1) * 0.01 for _ in range((X+1)*Y)]
		for y in range(Y):
			CONNECTIONS = 8 + (randint(0,5))
			for i in range(CONNECTIONS):
				p[y*(X+1) + randint(0,X)] = rnd()
			p[y*(X+1)+(X-1)+1] = 2*rnd()
		#
		self.insts = self.insts[:apres] + [   1  ] + self.insts[apres:]
		self.Y     = self.Y    [:apres] + [   Y  ] + self.Y    [apres:]
		self.poids = self.poids[:apres] + [len(p)] + self.poids[apres:]
		self.p     = self.p    [:apres] + [   p  ] + self.p    [apres:]

	def ecrire(self, fichier):
		with open(fichier, "wb") as co:
			co.write(st.pack('I'*C, *self.Y))
			co.write(st.pack('I'*C, *self.insts))
			#
			for ema_int in self.ema_int:
				co.write(st.pack(
					'I'*5,
					ema_int['source'],
					ema_int['nature'],
					ema_int['K_ema'],
					ema_int['intervalle'],
					ema_int['decale']
				))
				co.write(st.pack('I'*MAX_PARAMS, *ema_int['params']))
			#
			for i in range(len(self.p)):
				co.write(st.pack('f'*self.poids[i], *self.p[i]))

	def __call__(self):
		assert len(I__sources[0]) != 0
		assert len(I__sources[0]) == len(I__sources[1]) == len(I__sources[2]) == len(I__sources[3])

		with open("communication.bin", "wb") as co:
			co.write(st.pack('I'*len(self.Y), *self.Y))
			co.write(st.pack('I'*len(self.insts), *self.insts))
			#
			co.write(st.pack('I', I_PRIXS))
			#
			co.write(st.pack('I'*BLOQUES, *list(map(lambda x:x['intervalle'], self.ema_int))))
			co.write(st.pack('I'*BLOQUES, *list(map(lambda x:x['decale'],     self.ema_int))))
			type_norme = []
			min_norme, max_norme = [], []
			for _, n in self.lignes:
				if n != None:
					type_norme += [1]
					min_norme  += [n[0]]
					max_norme  += [n[1]]
				else:
					type_norme += [0]
					min_norme  += [0]
					max_norme  += [0]
			#
			for ligne,_ in self.lignes:
				co.write(st.pack('f'*I_PRIXS, *ligne))
			#
			for poids in self.p:
				co.write(st.pack('I', len(poids)))
				co.write(st.pack('f'*len(poids), *poids))
		#
		#system("gdb --args ./prog4__simple_mdl_pour_python communication.bin")
		system("./prog4__simple_mdl_pour_python communication.bin")
		#
		with open("communication.bin", "rb") as co:
			bins = co.read()
			ret, bins = lire_flotants(I_PRIXS-DEPART, bins)

		return ret