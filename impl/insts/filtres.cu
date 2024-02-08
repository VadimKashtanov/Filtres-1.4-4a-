#include "filtres_prixs.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_filtre_prixs(Mdl_t * mdl, uint c)
{
	mdl->inst_POIDS        [c] = BLOQUES*F_PAR_BLOQUES*N;
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = 2*mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);
	FOR(0, i, mdl->inst_POIDS[c])
		mdl->p[c][i] = (2*rnd()-1) * 1.0;
};

void plume_filtre_prixs(Mdl_t * mdl, uint c)
{
	printf("POIDS FILTRES: \n");
	FOR(0, b, BLOQUES) {
		FOR(0, f, F_PAR_BLOQUES) {
			printf("bloque=%i f=%i :", b, f);
			FOR(0, i, N)
				printf("%+f, ", mdl->p[c][b*F_PAR_BLOQUES*N + f*N + i]);
			printf("\n");
		}
	}
};

static float filtre(float * x, float * dif_x, float * f, float * locd) {
	float s = 0, d = 0;
	float f_nouveau = f[0];
	s += sqrtf(1 + fabs(x[0] - f_nouveau));
	float f_avant   = f_nouveau;
	FOR(1, i, N) {
		f_nouveau = f[i];
		s += sqrtf(1 + fabs(  x[i]   -   f_nouveau  ));
		d += powf((1 + fabs(dif_x[i] - (f_nouveau-f_avant))), 2);
		f_avant   = f_nouveau;
	};

	s = s/8-1;
	d = d/7-1;

	float y = expf(-s*s -d*d);

	locd[0] = -2*2*s*y;
	locd[1] = -2*2*d*y;

	return 2*y-1;
};

void intel_filtres_prixs___naive(
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * decales,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd)
{
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				y[(0+t)*bloques*f_par_bloque + b*f_par_bloque + _f] = filtre(
						x + b*PRIXS*N_FLTR + (depart+t-decales[b])*N_FLTR,
					dif_x + b*PRIXS*N_FLTR + (depart+t-decales[b])*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N,
					locd  + (0+t)*(bloques*f_par_bloque*2) + b*(f_par_bloque*2) + _f*2
				);
			}
		}
	}
};

static void d_filtre(float * x, float * dif_x, float * f, float * locd, float * dy, float * df) {
	float ds = locd[0] * dy[0] / 8;
	float dd = locd[1] * dy[0] / 7;
	//
	FOR(1, i, N)
	{
		//s += sqrtf(1 + fabs(  x[i]   -   f[i]  ));
		df[i] += ds * 1 / (2*sqrtf(1 + fabs(x[i] - f[i]))) * (-1) * signe(x[i] - f[i]);
		//d += powf((1 + fabs(dif_x[i] - dif_f[i])), 2);
		df[ i ] += dd * 2 * (1 + fabs(dif_x[i] - (f[i]-f[i-1]))) * signe(dif_x[i] - (f[i]-f[i-1])) * (-1);
		df[i-1] += dd * 2 * (1 + fabs(dif_x[i] - (f[i]-f[i-1]))) * signe(dif_x[i] - (f[i]-f[i-1])) * (+1);
	}
	df[0] += ds * 1 / (2*sqrtf(1 + fabs(x[0] - f[0]))) * (-1) * signe(x[0] - f[0]);
};

void  d_intel_filtres_prixs___naive(
	uint X_vars, uint Y_vars,
	uint depart, uint T,
	uint bloques, uint f_par_bloque, uint * decales,
	float * x, float * dif_x,
	float * f,
	float * y,
	float * locd,
	float * dy,
	float * df)
{
	FOR(0, t, T) {
		FOR(0, b, bloques) {
			FOR(0, _f, f_par_bloque) {
				d_filtre(
						x + b*PRIXS*N_FLTR + (depart+t-decales[b])*N_FLTR,
					dif_x + b*PRIXS*N_FLTR + (depart+t-decales[b])*N_FLTR,
					f     + b*f_par_bloque*N     + _f*N,
					locd  + (     0+t)*(bloques*f_par_bloque*2) + b*(f_par_bloque*2) + _f*2,
					dy    + (     0+t)*(bloques*f_par_bloque  ) + b*(f_par_bloque  ) + _f,
					df    + b*f_par_bloque*N     + _f*N
				);
			}
		}
	}
};

void f_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint depart = t0;
	uint X_vars=0, Y_vars=mdl->inst_VARS[inst];
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	if (mode == 0) {
		intel_filtres_prixs___naive(
			X_vars, Y_vars,
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->decales,
			mdl->normalisee, mdl->dif_normalisee,
			mdl->p[inst],
			mdl->y[inst],
			mdl->l[inst]);
	} else if (mode == 1/* || mode == 2 || mode == 3*/) {
		nvidia_filtres_prixs___naive(
			X_vars, Y_vars,
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->decales__d,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 2 || mode == 3) {
		nvidia_filtres_prixs___shared(
			X_vars, Y_vars,
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->decales__d,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst]);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	};
};

void df_filtres_prixs(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint depart = t0;
	uint X_vars=0, Y_vars=mdl->inst_VARS[inst];
	uint T = (t1-t0);
	ASSERT(T == mdl->T);
	if (mode == 0) {
		d_intel_filtres_prixs___naive(
			X_vars, Y_vars,
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->decales,
			mdl->normalisee, mdl->dif_normalisee,
			mdl->p[inst],
			mdl->y[inst],
			mdl->l[inst],
			mdl->dy[inst],
			mdl->dp[inst]);
	} else if (mode == 1/* || mode == 2 || mode == 3*/) {
		d_nvidia_filtres_prixs___naive(
			X_vars, Y_vars,
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->decales__d,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dp__d[inst]);
	} else if (mode == 2 || mode == 3) {
		d_nvidia_filtres_prixs___shared(
			X_vars, Y_vars,
			depart, T,
			BLOQUES, F_PAR_BLOQUES, mdl->decales__d,
			mdl->normalisee__d, mdl->dif_normalisee__d,
			mdl->p__d[inst],
			mdl->y__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour mes filtres", mode);
	}
};