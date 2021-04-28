/* single precision krls 
JunKyu Lee
First Coding : June. 2018
*/
#include <math.h>
#include <mkl.h> //should be replaced to openblas

void sdifference(float* x, float* u, float* sout, int m_t, int NIN)
{
	for(int i=0; i< m_t; i++) {
		for(int j=0; j<NIN; j++) sout[i*NIN+j] =x[j+i*NIN] - u[j];
	}
};

void sinnprods(float* x, float* sout, int m_t, int NIN)
{
	float* x_tmp = (float*)malloc(NIN*sizeof(float)); 
	for (int i=0; i<m_t; i++) {
		for(int j=0; j<NIN; j++) x_tmp[j] = x[NIN*i+j];
		sout[i] = cblas_sdot(NIN, x_tmp, 1, x_tmp, 1);	
	};
};

void sk_expo(float* x, float* y, float beta, int m_t)
{
	float mgpar;
	mgpar = -2.0*(beta*beta);	
	for(int i=0; i< m_t; i++) { y[i] = expf(x[i]/mgpar);}
};

float sald_delta(float* x1, float* x2, int m_t)
{
	float tmp;
	tmp = 1.0 - cblas_sdot(m_t, x1, 1, x2, 1);	
	return tmp;
};

void sinference_krls(float* x, float* dict_in, float* alpha, float beta, int m, float est_out[1], int NIN) 
{
	float* k_vec = (float*) malloc(MAXDICT*sizeof(float));
	float* diffs = (float*) malloc(NIN*MAXDICT*sizeof(float));
	float* sout = (float*) malloc(MAXDICT*sizeof(float));

	sdifference(dict_in, x, diffs, m, NIN);
	sinnprods(diffs, sout, m, NIN);
	sk_expo(sout, k_vec, beta, m);
	est_out[0] = cblas_sdot(m, k_vec, 1, alpha, 1);	
			
}; //inference krls function end



void single_krls(float beta, float nu, float regul, float* x, float y, float* dict_out, float* alpha_out, int m_out[1], float err_out[1], int init[1], int NIN) 
{
	float* k_vec = (float*) malloc(MAXDICT*sizeof(float));
	static float* K_inv = (float*) malloc(MAXDICT*MAXDICT*sizeof(float));
	float* tmp_K_inv = (float*) malloc(MAXDICT*MAXDICT*sizeof(float));
	static float* P = (float*) malloc(MAXDICT*MAXDICT*sizeof(float));
	float* tmp_P = (float*) malloc(MAXDICT*MAXDICT*sizeof(float));
	float* sout = (float*) malloc(MAXDICT*sizeof(float));
	float* a_vec = (float*) malloc(MAXDICT*sizeof(float));
	float* p_vec = (float*) malloc(MAXDICT*sizeof(float));
	float* q_vec = (float*) malloc(MAXDICT*sizeof(float));
	float* diffs = (float*) malloc(NIN*MAXDICT*sizeof(float));
	float delta_t = 0.0;
	float q_coef;
	float* a_over_del = (float*) malloc(MAXDICT*sizeof(float));
	int m_t; 
	
	static float* dict = (float*) malloc(NIN*MAXDICT*sizeof(float));
	static float* alpha = (float*) malloc(MAXDICT*sizeof(float));
	static int m[1];
	float err;
	
    //omp_set_num_threads(1);
		
	if(init[0]) {	
		for (int i=0; i<MAXDICT*MAXDICT; i++) { P[i] = 0.0;};
		for (int i=0; i<MAXDICT; i++) { alpha[i] = 0.0;};
		for (int i=0; i < NIN; i++) {dict[i] = x[i];}
		K_inv[0] = 1.0/(1+regul);
		alpha[0] = y*K_inv[0];
		m[0] = 1;
		init[0] = 0;
		m_t = m[0];
		P[0] = 1.0;
	}
	else {	
		m_t = m[0];
		sdifference(dict, x, diffs, m[0], NIN);
		sinnprods(diffs, sout, m[0], NIN);
		sk_expo(sout, k_vec, beta, m[0]);
		err = y-cblas_sdot(m[0], k_vec, 1, alpha, 1);
		err_out[0] = err;
		cblas_sgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], 1.0, K_inv, m[0], k_vec, 1, 0.0, a_vec, 1); //dot products  vector calculation
		delta_t = 1.0 - cblas_sdot(m[0], k_vec, 1, a_vec, 1) + regul;	
		
		if (delta_t > nu) {
			m_t = m[0];
			for (int i=0; i < NIN; i++) {dict[(m[0])*NIN + i] = x[i];}
			for (int i=0; i< (m[0]); i++) {a_over_del[i] = a_vec[i]/delta_t;}
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, (m[0]), (m[0]), 1, 1.0, a_over_del, m[0], a_vec, 1, 1.0, K_inv, m[0]);
			
			//Matrix rearrangement	
			for (int i=0; i<(m[0]); i++) { 
				for (int j=0; j<(m[0]); j++) {
					tmp_K_inv[(m[0]+1)*i+j] = K_inv[(m[0])*i+j];
					tmp_P[(m[0]+1)*i+j] = P[(m[0])*i+j];	
				}
			}
			
			for (int i=0; i<(m_t); i++) { 
				tmp_K_inv[(m[0]+1)*i+m[0]] = -1.0*a_over_del[i];
				tmp_K_inv[(m[0]+1)*m[0]+i] = -1.0*a_over_del[i];
				tmp_P[(m[0]+1)*i+m[0]] = 0.0;
				tmp_P[(m[0]+1)*m[0]+i] = 0.0;
			}
			
			m[0] = m[0] + 1;	
			m_t=m[0]; 
			tmp_K_inv[(m[0])*(m[0])-1] = 1.0/delta_t;
			tmp_P[(m[0])*(m[0])-1] = 1.0;
			
			for (int i=0; i<(m[0]); i++){
				for (int j=0; j<(m[0]); j++){
					K_inv[i*m[0] +j] = tmp_K_inv[i*m[0] +j]; P[i*m[0] +j] = tmp_P[i*m[0] +j];
				}
			}
					
			for (int i=0; i<m[0]-1; i++) alpha[i] = alpha[i] - a_over_del[i]*err;
			alpha[m[0]-1] = (1.0/delta_t)*err;	 
		}
		else { 
			m_t = m[0];
			cblas_sgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], 1.0, P, m[0], a_vec, 1, 0.0, p_vec, 1); //dot products  vector calculation
			q_coef = 1.0/(1.0+ cblas_sdot(m[0], a_vec, 1, p_vec, 1));	
			for(int i=0; i<m[0]; i++) q_vec[i] = q_coef*p_vec[i];			
		//	cblas_sgemm(CblasRowMajor,CblasTrans, CblasTrans, m[0], m[0], 1, -1.0, q_vec, m[0], p_vec, 1, 1.0, P, m[0]);
		cblas_sger (CblasRowMajor, m[0], m[0], -1.0, q_vec, 1, p_vec, 1, P, m[0]);
		
		cblas_sgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], err, K_inv, m[0], q_vec, 1, 1.0, alpha, 1); //alpha updates
		
		}; //delta condition end
	
	}; //init condition end
	
	m_out[0] = m[0];
	for(int i=0; i<m_t; i++) alpha_out[i] = alpha[i];
	for(int i=0; i<m_t; i++) {
		for(int j=0; j<NIN; j++) dict_out[i*NIN+j] = dict[i*NIN+j];
	};
	
}; 