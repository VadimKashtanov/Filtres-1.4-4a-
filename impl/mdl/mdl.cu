#include "mdl.cuh"

#include "filtres_prixs.cuh"
#include "dot1d.cuh"
#include "lstm1d.cuh"
#include "dot1d_blk.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

mdl_inst_f cree_inst[INSTS] = {
	cree_filtre_prixs,
	cree_dot1d,
	cree_dot1d_blk
};

mdl_f_f inst_f [INSTS] = {
	f_filtres_prixs,
	f_dot1d,
	f_dot1d_blk
};

mdl_f_f inst_df[INSTS] = {
	df_filtres_prixs,
	df_dot1d,
	df_dot1d_blk
};

char * nom_inst[INSTS] = {
	"filtres_prixs",
	"dot1d        ",
	"dot1d bloques"
};

mdl_inst_f plume_inst[INSTS] = {
	plume_filtre_prixs,
	plume_dot1d,
	plume_dot1d_blk
};

static void calculer_normalisee__et__dif_normalisee(Mdl_t * mdl) { 
	FOR(0, b, BLOQUES) {
		FOR(DEPART-MAX_DECALES, t, FIN) {
			//	_max & _min pour ce filtre-8
			float _max, _min;
			if (mdl->type_de_norme[b] == NORME_CLASSIQUE) {
				_max = mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle];
				_min = mdl->bloque[b]->brute[t - 0*mdl->bloque[b]->intervalle];
				FOR(1, i, N_FLTR) {
					if (_max < mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle])
						_max = mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle];
					if (_min > mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle])
						_min = mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle];
				}
			} else if (mdl->type_de_norme[b] == NORME_THEORIQUE) {
				_min = mdl->min_theorique[b];
				_max = mdl->max_theorique[b];
			} else {
				ERR("Norme ni classique, ni theorique, mdl->type_de_norme[b] == %i", mdl->type_de_norme[b]);
			};

			//
			FOR(0, i, N_FLTR) {
				mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] = ( mdl->bloque[b]->brute[t-i*mdl->bloque[b]->intervalle] - _min)/( _max - _min );
				//if (mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i]==0) printf("%i %i %i\n", b,t,i);
			}
		};

		FOR(DEPART-MAX_DECALES, t, FIN) {
			FOR(1, i, N_FLTR)
				mdl->dif_normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] = mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i] - mdl->normalisee[b*PRIXS*N_FLTR+t*N_FLTR+i-1];
			mdl->dif_normalisee[b*PRIXS*N_FLTR+t*N_FLTR+N_FLTR+0] = 0.f;
		}
	}

	mdl->normalisee__d     = cpu_vers_gpu<float>(mdl->normalisee,     BLOQUES * PRIXS * N_FLTR);
	mdl->dif_normalisee__d = cpu_vers_gpu<float>(mdl->dif_normalisee, BLOQUES * PRIXS * N_FLTR);
};

static uint tout_zeroiser[C] = UNIFORME_C(1);

Mdl_t * cree_mdl(
	uint T,
	uint Y[C],
	uint insts[C],
	ema_int_t * bloque[BLOQUES]
) {
	ASSERT(Y[C-1] == P);
	ASSERT(Y[ 0 ] == BLOQUES * F_PAR_BLOQUES);
	ASSERT(insts[C-1] == DOT1D);				//	Afin d'assurer un Y=inst_VARS
	
	Mdl_t * mdl = alloc<Mdl_t>(1);

	mdl->T = T;

	//
	FOR(0, i, BLOQUES) {
		mdl->bloque[i]  = bloque[i];
		mdl->decales[i] = bloque[i]->decale;
		mdl->intervalles[i] = bloque[i]->intervalle;
		//
		mdl->type_de_norme[i] = bloque[i]->type_de_norme;
		mdl->min_theorique[i] = bloque[i]->min_theorique;
		mdl->max_theorique[i] = bloque[i]->max_theorique;
	};

	mdl->decales__d = cpu_vers_gpu<uint>(mdl->decales, BLOQUES);
	mdl->intervalles__d = cpu_vers_gpu<uint>(mdl->intervalles, BLOQUES);

	//
	calculer_normalisee__et__dif_normalisee(mdl);
	//raise(SIGINT);

	//	Architecture
	memcpy(mdl->insts, insts, sizeof(uint) * C);
	memcpy(mdl->Y,         Y, sizeof(uint) * C);

	//	Allocation
	mdl->total_POIDS = 0;
	FOR(0, c, C) {
		if (c>0) ASSERT(insts[c] != 0);
		ASSERT(Y[c] <= MAX_Y);
		//
		cree_inst[insts[c]](mdl, c);
		//
		//mdl->p [c] = alloc<float>(mdl->inst_POIDS[c]);
		mdl->y [c] = alloc<float>(mdl->inst_VARS [c] * T);
		mdl->l [c] = alloc<float>(mdl->inst_LOCDS[c] * T);
		mdl->dy[c] = alloc<float>(mdl->inst_VARS [c] * T);
		mdl->dp[c] = alloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->p__d [c] = cpu_vers_gpu<float>(mdl->p[c], mdl->inst_POIDS[c]);
		mdl->y__d [c] = cudalloc<float>(mdl->inst_VARS [c] * T);
		mdl->l__d [c] = cudalloc<float>(mdl->inst_LOCDS[c] * T);
		mdl->dy__d[c] = cudalloc<float>(mdl->inst_VARS [c] * T);
		mdl->dp__d[c] = cudalloc<float>(mdl->inst_POIDS[c]);
		//
		mdl->total_POIDS += mdl->inst_POIDS[c];
	}
	ASSERT(mdl->inst_DEPART_SORTIE[C-1] == 0);
	//
	mdl_normer_les_filtres(mdl);
	//
	mdl_zero_deriv_gpu(mdl, tout_zeroiser);
	mdl_zero_deriv_cpu(mdl, tout_zeroiser);
	//
	return mdl;
};

void mdl_normer_les_filtres(Mdl_t * mdl) {
	FOR(0, b, BLOQUES) {
		FOR(0, f, F_PAR_BLOQUES) {
			float min, max;
			if (mdl->type_de_norme[b] == NORME_CLASSIQUE) {
				max=mdl->p[0][b*F_PAR_BLOQUES*N + f*N+0];
				min=mdl->p[0][b*F_PAR_BLOQUES*N + f*N+0];
				FOR(1, i, N) {
					if (max < mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]) max = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
					if (min > mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]) min = mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i];
				}
			} else if (mdl->type_de_norme[b] == NORME_THEORIQUE) {
				min = 0.0;	//se sont des filtres
				max = 1.0;	//pas les natures
			} else {
				ERR("mdl->type_de_norme[b]=%i", mdl->type_de_norme[b]);
			}
			//
			FOR(0, i, N) mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i] = (mdl->p[0][b*F_PAR_BLOQUES*N + f*N+i]-min)/(max-min);
		};
	}
	CONTROLE_CUDA(cudaMemcpy(mdl->p__d[0], mdl->p[0], sizeof(float)*BLOQUES*F_PAR_BLOQUES*N, cudaMemcpyHostToDevice))
};

PAS_OPTIMISER()
void mdl_verif(Mdl_t * mdl) {
	FOR(1, c, C) {
		float * r = gpu_vers_cpu<float>(mdl->p__d[c], mdl->inst_POIDS[c]);
		FOR(0, i, mdl->inst_POIDS[c]) ASSERT(fabs(r[i]-mdl->p[c][i]) < 0.01);
		free(r);
	}
};

PAS_OPTIMISER()
void mdl_gpu_vers_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p[c],  mdl->p__d[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->y[c],  mdl->y__d[c],  sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->l[c],  mdl->l__d[c],  sizeof(float)*mdl->inst_LOCDS[c]*mdl->T, cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy[c], mdl->dy__d[c], sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyDeviceToHost));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp[c], mdl->dp__d[c], sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyDeviceToHost));
	}
}

PAS_OPTIMISER()
void mdl_cpu_vers_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemcpy(mdl->p__d[c],  mdl->p[c],  sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->y__d[c],  mdl->y[c],  sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->l__d[c],  mdl->l[c],  sizeof(float)*mdl->inst_LOCDS[c]*mdl->T, cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dy__d[c], mdl->dy[c], sizeof(float)*mdl->inst_VARS[c]*mdl->T,  cudaMemcpyHostToDevice));
		CONTROLE_CUDA(cudaMemcpy(mdl->dp__d[c], mdl->dp[c], sizeof(float)*mdl->inst_POIDS[c],       cudaMemcpyHostToDevice));
	}
};

PAS_OPTIMISER()
void liberer_mdl(Mdl_t * mdl) {
	CONTROLE_CUDA(cudaFree(mdl->decales__d));
	FOR(0, c, C) {
		free(mdl->p [c]);
		free(mdl->y [c]);
		free(mdl->l [c]);
		free(mdl->dy[c]);
		free(mdl->dp[c]);
		//
		CONTROLE_CUDA(cudaFree(mdl->p__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->y__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->l__d [c]));
		CONTROLE_CUDA(cudaFree(mdl->dy__d[c]));
		CONTROLE_CUDA(cudaFree(mdl->dp__d[c]));
	}
};

PAS_OPTIMISER()
void mdl_zero_cpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		memset(mdl->y [c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T);
	}
};

PAS_OPTIMISER()
void mdl_zero_gpu(Mdl_t * mdl) {
	FOR(0, c, C) {
		CONTROLE_CUDA(cudaMemset(mdl->y__d [c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T));
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_cpu(Mdl_t * mdl, uint zeroiser[C]) {
	FOR(0, c, C) {
		if (zeroiser[c]) {
			memset(mdl->dy[c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T);
			memset(mdl->dp[c], 0, sizeof(float) * mdl->inst_POIDS[c]);
		}
	}
};

PAS_OPTIMISER()
void mdl_zero_deriv_gpu(Mdl_t * mdl, uint zeroiser[C]) {
	FOR(0, c, C) {
		if (zeroiser[c]) {
			CONTROLE_CUDA(cudaMemset(mdl->dy__d[c], 0, sizeof(float) * mdl->inst_VARS [c] * mdl->T));
			CONTROLE_CUDA(cudaMemset(mdl->dp__d[c], 0, sizeof(float) * mdl->inst_POIDS[c]));
		}
	}
};