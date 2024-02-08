#! /usr/bin/python3

from mdl import *

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: (x if abs(x) >= 0.0 else 0)

prixs = I__sources[0]

if __name__ == "__main__":
	mdl = Mdl("mdl.bin")

	print("Calcule ...")
	pred = mdl()
	print("Fin Calcule")

	_prixs = I__sources[0][DEPART:]

	print(len(pred), len(_prixs))

	plt.plot(e_norme(_prixs));
	plt.plot(pred, 'o');
	#
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)): plt.plot([i for _ in pred], e_norme(list(range(len(pred)))), '--')

	plt.show()

	##	================ Gain ===============
	DEPART = (N+MAX_DECALES)*MAX_INTERVALLES
	#
	u = 40
	usd = []
	#
	decale = 0
	LEVIER = 25
	POURCENT = 1.0
	#
	#	Diagnostique
	gain_sur_achat  = 0
	gain_sur_vente  = 0
	perte_sur_achat = 0
	perte_sur_vente = 0
	#
	preds = []
	for i in range(DEPART, I_PRIXS-1):
		__pred = -pred[i-DEPART]
		preds += [__pred]
		u += u * LEVIER * POURCENT * (__pred) * (prixs[i+1]/prixs[i]-1)
		#
		#print(__pred)
		if signe(__pred) == signe((prixs[i+1]/prixs[i]-1)):
			if signe(__pred) >= 0:
				gain_sur_achat += 1
			else:
				gain_sur_vente += 1 
		else:
			if signe(__pred) >= 0:
				perte_sur_achat += 1
			else:
				perte_sur_vente += 1
		#
		if (u <= 0): u = 0
		print(f"usd = {u}")
		usd += [u]
	print(f"Gain  sur achat={gain_sur_achat}, Gain  sur vente={gain_sur_vente}")
	print(f"Perte sur achat={perte_sur_achat}, Perte sur vente={perte_sur_vente}")
	plt.plot(usd)
	#plt.plot([max(usd) * i for i in preds])
	plt.show()