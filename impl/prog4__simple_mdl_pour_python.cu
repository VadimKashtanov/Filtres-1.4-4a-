#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static float filtre(uint depart, float * x, float * f, uint intervalle, uint decale, uint type_norme, float _min, float _max) {
	float normer_x[N];
	//
	FOR(0, i, N) normer_x[i] = x[depart - (decale+i)*intervalle];
	//
	if (type_norme == NORME_CLASSIQUE) {
		_min=normer_x[0];
		_max=normer_x[0];
		//
		FOR(1, i, N) {
			float a = normer_x[i];
			if (a > _max) _max = a;
			if (a < _min) _min = a;
		}
	} else if (type_norme == NORME_THEORIQUE) {
		// rien
	} else {
		ERR("type_norme == %i", type_norme);
	}
	//
	FOR(0, i, N) normer_x[i] = (normer_x[i]-_min)/(_max-_min);
	//
	float s = 0, d = 0;
	float f_nouveau = f[0];
	float x_nouveau = normer_x[0];
	s += sqrtf(1 + fabs(x_nouveau - f_nouveau));
	float f_avant = f_nouveau;
	float x_avant = x_nouveau;
	FOR(1, i, N) {
		f_nouveau = f[i];
		x_nouveau = normer_x[i];
		s += sqrtf(1 + fabs(  x_nouveau   -   f_nouveau  ));
		d += powf((1 + fabs((x_nouveau-x_avant) - (f_nouveau-f_avant))), 2);
		f_avant   = f_nouveau;
		x_avant   = x_nouveau;
	};

	s = s/8-1;
	d = d/7-1;

	return 2*expf(-s*s -d*d)-1;
};


int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	//
	FILE * fp = fopen(argv[1], "rb");
	//
	uint Y[C];
	FREAD(Y, sizeof(uint), C, fp);
	uint insts[C];
	FREAD(insts, sizeof(uint), C, fp);
	//
	//
	//
	//
	uint PRIXS_bitget;
	FREAD(&PRIXS_bitget, sizeof(uint), 1, fp);
	uint intervalles[BLOQUES], decales[BLOQUES];
	FREAD(intervalles, sizeof(uint), BLOQUES, fp);
	FREAD(decales,     sizeof(uint), BLOQUES, fp);
	//
	//
	//
	uint type_norme[BLOQUES];
	float _min[BLOQUES], _max[BLOQUES];
	FREAD(type_norme, sizeof(uint), BLOQUES, fp);
	FREAD(_min,       sizeof(float), BLOQUES, fp);
	FREAD(_max,       sizeof(float), BLOQUES, fp);
	//
	//
	//
	float * lignes = alloc<float>(PRIXS_bitget*BLOQUES);
	FREAD(lignes, sizeof(float), PRIXS_bitget*BLOQUES, fp);
	//
	float * poids[C];
	FOR(0, c, C) {
		uint POIDS;
		FREAD(&POIDS, sizeof(uint), 1, fp);
		poids[c] = alloc<float>(POIDS);
		FREAD(poids[c], sizeof(float), POIDS, fp);
	}
	//
	fclose(fp);

	//	------------- Calcule ----------------
	float * y_avant   = alloc<float>( PRIXS_bitget*MAX_Y );
	float * y_nouveau = alloc<float>( PRIXS_bitget*MAX_Y );
	//
	FOR(0, f, BLOQUES*F_PAR_BLOQUES) {
		uint b = (f - (f % F_PAR_BLOQUES)) / F_PAR_BLOQUES;
		FOR(DEPART, t, PRIXS_bitget) {
			y_nouveau[t*MAX_Y + f] = filtre(
				b*PRIXS_bitget + t,
				lignes,
				poids[0] + f*N,
				intervalles[b], decales[b],
				type_norme[b],
				_min[b], _max[b]
			);
		}
	};
	FOR(0, i, PRIXS_bitget*MAX_Y) y_avant[i] = y_nouveau[i];
	//
	FOR(1, c, C) {
		if (insts[c] == DOT1D) {
			uint X = Y[c-1];
			FOR(0, i, Y[c]) {
				FOR(DEPART, t, PRIXS_bitget) {
					float s = poids[c][(X+1)*i + X-1+1];
					FOR(0, j, X) s += poids[c][(X+1)*i + j] * y_avant[t*MAX_Y + j];
					y_nouveau[t*MAX_Y + i] = tanh(s);
				};
			};
		} else if (insts[c] == DOT1D_BLK) {
#include "dot1d_blk.cuh"
			uint  X = Y[c-1];
			uint _Y = Y[ c ];
			//
			uint X_blk =  X / DOT1D_BLK_BLOQUES;
			uint Y_blk = _Y / DOT1D_BLK_BLOQUES;
			uint P_blk =  ( X_blk + 1 ) * Y_blk;
			//
			FOR(DEPART, t, PRIXS_bitget) {
				FOR(0, blk, DOT1D_BLK_BLOQUES) {
					//
					uint depart_y = blk * Y_blk;
					uint depart_x = blk * X_blk;
					uint depart_p = blk * P_blk;
					//
					FOR(0, y, Y_blk) {
						float s = poids[c][depart_p + (X_blk+1)*y + (X_blk+1)+1];
						FOR(0, j, X_blk)
							s += poids[c][depart_p + (X_blk+1)*y] * y_avant[t*MAX_Y + depart_x + j];
						y_nouveau[t*MAX_Y + depart_y + y] = tanh(s);
					};
				};
			}
		} else {
			ERR("Inst = %i", insts[c]);
		}

		/*#pragma omp parallel
		#pragma omp for*/
		FOR(0, i, PRIXS_bitget*MAX_Y) y_avant[i] = y_nouveau[i];
	};

	//	---------- Ecrire Resultat ----------
	fp = fopen(argv[1], "wb");
	//
	float res[PRIXS_bitget];
	FOR(DEPART, t, PRIXS_bitget) res[t] = y_nouveau[t*MAX_Y + 0];
	FWRITE(res+DEPART, sizeof(float), (PRIXS_bitget-DEPART), fp);
	//
	fclose(fp);
}