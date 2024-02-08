#include "main.cuh"

/*
##	1) mdl norme theorique (borne) max et min pas du filtre, mais constant
##	2) mdl T bloque (pas *PRIXS) mdl_cree(T=16*16*1)
	3) plus de filtres
	4) filtres (d=1, decale=(0,4,8,16)), (d=32, decale=(0,8)), (d=256, decale=(0))
	5) dot1d_bloque 8172 -> 4*8172 -> 2*8172 -> 1*8172 -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
	6) 256 -> 512 -> 512 -> 256 -> 512 -> 512 -> 256 -> ...
*/

#include "../impl_tmpl/tmpl_etc.cu"

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	uint fois = (t1-t0)/mdl->T;
	//
	float moyenne[P] = {0};
	//
	FOR(0, i, fois) {
		float * ancien = mdl_pred(mdl, t0 + i*mdl->T, (i+1)*mdl->T, 3);
		FOR(0, p, P) moyenne[p] += ancien[p];
		free(ancien);
	}
	printf("PRED GENERALE = ");
	FOR(0, p, P) printf(" %f%% ", 100*moyenne[p]/(float)fois);
	printf("\n");
};

float pourcent_masque_nulle[C] = {0};

/*float pourcent_masque[C] = {
	.10,
	.10,
	.10,
	.10,
	.10,
	.10,
	.10,
	.10,
	.10,
	.10,
	.00
};*/

float * alpha = de_a(1e-4, 1e-4, C);

uint optimiser_tous_les[C] = UNIFORME_C(1);

#define GRAND_T (16*16*1)

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	alpha[0] = 1e-2;
	
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");
	ecrire_structure_generale("structure_generale.bin");

	/*uint Y[C] = {
		2048,
		1024,
		512,
		256,
		128,
		64,
		32,
		16,
		8,
		4,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D
	};
	//
	//	Assurances :
	ema_int_t * bloque[BLOQUES] = {
	//			    Source,      Nature,  K_ema, Intervalle, decale,     {params}
	// ----
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 1, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 4, 32, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 8, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 16, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 16, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 8.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 64, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 16.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 128, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 32.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_PRIXS, DIRECT, 256, 256, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 1, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 4, 32, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 8, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 16, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 16, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 8.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 64, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 16.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 128, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 32.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_HIGH, DIRECT, 256, 256, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 1, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 4, 32, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 8, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 16, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 16, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 8.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 64, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 16.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 128, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 32.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_LOW, DIRECT, 256, 256, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 1, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 1, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 4, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 32, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 4, 32, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 1.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 1.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 8, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 8, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 8, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 2.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 2.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 16, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 16, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 16, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 8.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 8.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 64, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 64, 64, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 16.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 16.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 128, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 128, 128, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 32.0, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 32.0, 8, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 256, 0, cree_DIRECTE()),
		cree_ligne(SRC_VOLUMES, DIRECT, 256, 256, 8, cree_DIRECTE()),
	// ----
		cree_ligne(SRC_PRIXS, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 1, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 4, 4, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 1.0, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 16, 16, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 64, 4.0, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 64, 64, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 128, 8.0, 8, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_PRIXS, MACD, 128, 128, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 1, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 4, 4, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 1.0, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 16, 16, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 64, 4.0, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 64, 64, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 128, 8.0, 8, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_HIGH, MACD, 128, 128, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 1, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 4, 4, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 1.0, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 16, 16, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 64, 4.0, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 64, 64, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 128, 8.0, 8, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_LOW, MACD, 128, 128, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 1, 1, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 1, 1, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 4, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 4, 4, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 1.0, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 1.0, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 16, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 16, 16, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 4.0, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 4.0, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 64, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 64, 64, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 128, 8.0, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 128, 8.0, 8, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 128, 128, 0, cree_MACD(1)),
		cree_ligne(SRC_VOLUMES, MACD, 128, 128, 8, cree_MACD(1)),
	// ----
		cree_ligne(SRC_HIGH, CHIFFRE, 1, 1, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 1, 1, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 8, 8, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 8, 8, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 32, 32, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 32, 32, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 128, 128, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 128, 128, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 1, 1, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 1, 1, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 8, 8, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 8, 8, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 32, 32, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 32, 32, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 128, 128, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_HIGH, CHIFFRE, 128, 128, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 1, 1, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 1, 1, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 8, 8, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 8, 8, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 32, 32, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 32, 32, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 128, 128, 0, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 128, 128, 8, cree_CHIFFRE(1000)),
		cree_ligne(SRC_LOW, CHIFFRE, 1, 1, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 1, 1, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 8, 8, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 8, 8, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 32, 32, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 32, 32, 8, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 128, 128, 0, cree_CHIFFRE(10000)),
		cree_ligne(SRC_LOW, CHIFFRE, 128, 128, 8, cree_CHIFFRE(10000))
	};
	//
	Mdl_t * mdl = cree_mdl(GRAND_T, Y, insts, bloque);*/

	/*Mdl_t * mdl = ouvrire_mdl(GRAND_T, "mdl.bin");

	enregistrer_les_lignes_brute(mdl, "lignes_brute.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = t0 + ROND_MODULO((FIN-DEPART), (16*16));
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%(16*16)=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%(16*16));
	//
	plume_pred(mdl, t0, t1);
	//
	srand(time(NULL));
#define PERTURBATIONS 0
	//
	uint REP = 300;
	FOR(0, rep, REP) {
		perturber(mdl, 50);
		perturber_filtres(mdl, 50);
		optimisation_mini_packet(
			mdl,
			t0, t1, GRAND_T,
			alpha, 1.0,
			RMSPROP, 40,
			pourcent_masque,
			//pourcent_masque_nulle,
			PERTURBATIONS,
			optimiser_tous_les);
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		plume_pred(mdl, t0, t1);
		//
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, REP);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);*/

	//	-- Fin --
	liberer_tout();
};