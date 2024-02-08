prixs, haut, bas, volumes = "SRC_PRIXS", "SRC_HIGH", "SRC_LOW", "SRC_VOLUMES"

directes = "DIRECT", """
{'K': 1, 'interv': 1}
{'K': 1, 'interv': 1}
{'K': 1, 'interv': 8}
{'K': 4, 'interv': 4}
{'K': 4, 'interv': 4}
{'K': 4, 'interv': 32}
{'K': 8, 'interv': 1.0}
{'K': 8, 'interv': 1.0}
{'K': 8, 'interv': 8}
{'K': 8, 'interv': 64}
{'K': 16, 'interv': 2.0}
{'K': 16, 'interv': 2.0}
{'K': 16, 'interv': 16}
{'K': 16, 'interv': 128}
{'K': 64, 'interv': 8.0}
{'K': 64, 'interv': 64}
{'K': 128, 'interv': 16.0}
{'K': 128, 'interv': 128}
{'K': 256, 'interv': 32.0}
{'K': 256, 'interv': 256}
""", (prixs, haut, bas, volumes), ("cree_DIRECTE()",)

macds = "MACD", """
{'K': 1, 'interv': 1}
{'K': 4, 'interv': 4}
{'K': 16, 'interv': 1.0}
{'K': 16, 'interv': 16}
{'K': 64, 'interv': 4.0}
{'K': 64, 'interv': 64}
{'K': 128, 'interv': 8.0}
{'K': 128, 'interv': 128}
""", (prixs, haut, bas, volumes), ("cree_MACD(1)",)

chiffres = "CHIFFRE", """
{'K': 1, 'interv': 1}
{'K': 8, 'interv': 8}
{'K': 32, 'interv': 32}
{'K': 128, 'interv': 128}
""", (haut, bas), ("cree_CHIFFRE(1000)", "cree_CHIFFRE(10000)")

k = 0
for nom, lignes, sources, params in (directes, macds, chiffres):
	lignes = list(map(eval, lignes.strip('\n').split('\n')))
	print("\t// -------")
	for src in sources:
		for param in params:
			for i in lignes:
				for decale in (0, 8):
					print(f"\t\tcree_ligne({src}, {nom}, {i['K']}, {i['interv']}, {decale}, {param}),")
					k += 1

print(f"\nlignes = {k}")