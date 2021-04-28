/* single precision krls 
JunKyu Lee
First Coding : June. 2018
*/
#include <math.h>
#include <mkl.h> //should be replaced to openblas

void difference(double* x, double* u, double* sout, int m_t, int NIN)
{
	for(int i=0; i< m_t; i++) {
		for(int j=0; j<NIN; j++) sout[i*NIN+j] =x[j+i*NIN] - u[j];
	}
};

void innprods(double* x, double* sout, int m_t, int NIN)
{
	double* x_tmp = (double*)malloc(NIN*sizeof(double)); 
	for (int i=0; i<m_t; i++) {
		for(int j=0; j<NIN; j++) x_tmp[j] = x[NIN*i+j];
		sout[i] = cblas_ddot(NIN, x_tmp, 1, x_tmp, 1);	
	};
};

void k_expo(double* x, double* y, double beta, int m_t)
{
	double mgpar;
	mgpar = -2.0*(beta*beta);	
	for(int i=0; i< m_t; i++) { y[i] = exp(x[i]/mgpar);}
}; 

double ald_delta(double* x1, double* x2, int m_t)
{
	double tmp;
	tmp = 1.0 - cblas_ddot(m_t, x1, 1, x2, 1);	
	return tmp;
};

void inference_krls(double* x, double* dict_in, double* alpha, double beta, int m, double est_out[1], int NIN) 
{
	double* k_vec = (double*) malloc(MAXDICT*sizeof(double));
	double* diffs = (double*) malloc(NIN*MAXDICT*sizeof(double));
	double* sout = (double*) malloc(MAXDICT*sizeof(double));

	difference(dict_in, x, diffs, m, NIN);
	innprods(diffs, sout, m, NIN);
	k_expo(sout, k_vec, beta, m);
	est_out[0] = cblas_ddot(m, k_vec, 1, alpha, 1);	
			
}; //inference krls function end



void double_krls(double beta, double nu, double regul, double* x, double y, double* dict_out, double* alpha_out, int m_out[1], double err_out[1], int init[1], int NIN) 
{
	static int m[1];

	double* k_vec = (double*) malloc(m[0]*sizeof(double));
	static double K_inv[MAXDICT*MAXDICT];
	double* tmp_K_inv = (double*) malloc((m[0]+1)*(m[0]+1)*sizeof(double));
	static double P[MAXDICT*MAXDICT];
	double* tmp_P = (double*) malloc((m[0]+1)*(m[0]+1)*sizeof(double));
	double* sout = (double*) malloc(m[0]*sizeof(double));
	double* a_vec = (double*) malloc(m[0]*sizeof(double));
	double* p_vec = (double*) malloc(m[0]*sizeof(double));
	double* q_vec = (double*) malloc(m[0]*sizeof(double));
	double* diffs = (double*) malloc(NIN*m[0]*sizeof(double));
	double delta_t = 0.0;
	double q_coef;
	double* a_over_del = (double*) malloc(m[0]*sizeof(double));
	int m_t; 
	
	static double* dict = (double*) malloc(NIN*MAXDICT*sizeof(double));
	static double* alpha = (double*) malloc(MAXDICT*sizeof(double));
	
	double err;
	
    //omp_set_num_threads(1);
		
	if(init[0]) {	
//		for (int i=0; i<MAXDICT*MAXDICT; i++) { P[i] = 0.0;};
//		for (int i=0; i<MAXDICT; i++) { alpha[i] = 0.0;};
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
		difference(dict, x, diffs, m[0], NIN);
		innprods(diffs, sout, m[0], NIN);
		k_expo(sout, k_vec, beta, m[0]);
		err = y-cblas_ddot(m[0], k_vec, 1, alpha, 1);
		err_out[0] = err;
		cblas_dgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], 1.0, K_inv, m[0], k_vec, 1, 0.0, a_vec, 1); //dot products  vector calculation
		delta_t = 1.0 - cblas_ddot(m[0], k_vec, 1, a_vec, 1) + regul;	
		
		if (delta_t > nu) {
			m_t = m[0];
			for (int i=0; i < NIN; i++) {dict[(m[0])*NIN + i] = x[i];}
			for (int i=0; i< (m[0]); i++) {a_over_del[i] = a_vec[i]/delta_t;}
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, (m[0]), (m[0]), 1, 1.0, a_over_del, m[0], a_vec, 1, 1.0, K_inv, m[0]);
			
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
			cblas_dgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], 1.0, P, m[0], a_vec, 1, 0.0, p_vec, 1); //dot products  vector calculation
			q_coef = 1.0/(1.0+ cblas_ddot(m[0], a_vec, 1, p_vec, 1));	
			for(int i=0; i<m[0]; i++) q_vec[i] = q_coef*p_vec[i];			
			//cblas_dgemm(CblasRowMajor,CblasTrans, CblasTrans, m[0], m[0], 1, -1.0, q_vec, m[0], p_vec, 1, 1.0, P, m[0]);
			//void cblas_dger (const CBLAS_LAYOUT Layout, const MKL_INT m, const MKL_INT n, const double alpha, const double *x, const MKL_INT incx, const double *y, const MKL_INT incy, double *a, const MKL_INT lda)			
			cblas_dger (CblasRowMajor, m[0], m[0], -1.0, q_vec, 1, p_vec, 1, P, m[0]);
			cblas_dgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], err, K_inv, m[0], q_vec, 1, 1.0, alpha, 1); //alpha updates
		}; //delta condition end
	
	}; //init condition end
	
	m_out[0] = m[0];
	for(int i=0; i<m_t; i++) alpha_out[i] = alpha[i];
	for(int i=0; i<m_t; i++) {
		for(int j=0; j<NIN; j++) dict_out[i*NIN+j] = dict[i*NIN+j];
	};
	
}; 
