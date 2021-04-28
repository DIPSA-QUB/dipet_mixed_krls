/* single precision krls 
JunKyu Lee
First Coding : June. 2018
*/

void mixed_krls(double beta, double nu, double regul, double* x, double y, double* dict_out, double* alpha_out, int m_out[1], double err_out[1], int init[1], int NIN) 
{
	
	static int m[1] = {1};
	int m_temp, m_temp1; 
	m_temp= m[0];
	m_temp1 = m[0]+1;
	
	double* k_vec = (double*) malloc(m_temp*sizeof(double));
	static double K_inv [MAXDICT*MAXDICT];
	double* tmp_K_inv = (double*) malloc(m_temp1*m_temp1*sizeof(double));
	static float P [MAXDICT*MAXDICT];
	float* tmp_P = (float*) malloc(m_temp1*m_temp1*sizeof(float));
	double* sout = (double*) malloc(m_temp*sizeof(double));
	double* a_vec = (double*) malloc(m_temp*sizeof(double));
	float* sa_vec = (float*) malloc(m_temp*sizeof(float));
	float* p_vec = (float*) malloc(m_temp*sizeof(float));
	double* q_vec = (double*) malloc(m_temp*sizeof(double));
	float* sq_vec = (float*) malloc(m_temp*sizeof(float));
	double* diffs = (double*) malloc(NIN*m_temp*sizeof(double));
	double delta_t = 0.0;
	float q_coef;
	double* a_over_del = (double*) malloc(m_temp*sizeof(double));
	
	static double* dict = (double*) malloc(NIN*MAXDICT*sizeof(double));
	static double* alpha = (double*) malloc(MAXDICT*sizeof(double));
	
	double err;
	
    //omp_set_num_threads(1);
		
	if(init[0]) {	
	//	for (int i=0; i<m[0]*m[0]; i++) { P[i] = 0.0;};
	//	for (int i=0; i<m[0]; i++) { alpha[i] = 0.0;};
		for (int i=0; i < NIN; i++) {dict[i] = x[i];}
		K_inv[0] = 1.0/(1+regul);
		alpha[0] = y*K_inv[0];
		m[0] = 1;
		init[0] = 0;
		P[0] = 1.0;
	}
	else {	
		difference(dict, x, diffs, m[0], NIN);
		innprods(diffs, sout, m[0], NIN);
		k_expo(sout, k_vec, beta, m[0]);
		err = y-cblas_ddot(m[0], k_vec, 1, alpha, 1);
		err_out[0] = err;
		cblas_dgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], 1.0, K_inv, m[0], k_vec, 1, 0.0, a_vec, 1); //dot products  vector calculation
		delta_t = 1.0 - cblas_ddot(m[0], k_vec, 1, a_vec, 1) + regul;	
		
		if (delta_t > nu) {
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
			
			for (int i=0; i<m[0]; i++) { 
				tmp_K_inv[(m[0]+1)*i+m[0]] = -1.0*a_over_del[i];
				tmp_K_inv[(m[0]+1)*m[0]+i] = -1.0*a_over_del[i];
				tmp_P[(m[0]+1)*i+m[0]] = 0.0;
				tmp_P[(m[0]+1)*m[0]+i] = 0.0;
			}
			
			m[0] = m[0] + 1;	
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
			for(int i=0; i<m[0]; i++) sa_vec[i] = (float)a_vec[i];
			cblas_sgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], 1.0, P, m[0], sa_vec, 1, 0.0, p_vec, 1); //dot products  vector calculation
			q_coef = 1.0/(1.0+ cblas_sdot(m[0], sa_vec, 1, p_vec, 1));	
			for(int i=0; i<m[0]; i++) sq_vec[i] = q_coef*p_vec[i];			
			//cblas_sgemm(CblasRowMajor,CblasTrans, CblasTrans, m[0], m[0], 1, -1.0, sq_vec, m[0], p_vec, 1, 1.0, P, m[0]);
			cblas_sger (CblasRowMajor, m[0], m[0], -1.0, sq_vec, 1, p_vec, 1, P, m[0]);

			for(int i=0; i<m[0]; i++) q_vec[i] = (double)sq_vec[i];	
			cblas_dgemv(CblasRowMajor,CblasNoTrans, m[0], m[0], err, K_inv, m[0], q_vec, 1, 1.0, alpha, 1); //alpha updates
		
		/*
				    m_t = m[0];
			    for(int i=0; i<m_t; i++) sa_vec[i] = (float)a_vec[i];
				cblas_sgemv(CblasRowMajor,CblasNoTrans, m_t, m_t, 1.0, P, m_t, a_vec, 1, 0.0, p_vec, 1); //dot products  vector calculation
				
				q_coef = 1.0/(1.0+ cblas_sdot(m_t, a_vec, 1, p_vec, 1));	
				for(int i=0; i<m_t; i++) {q_vec[i] = q_coef*p_vec[i];}		
					
				cblas_sgemm(CblasRowMajor,CblasTrans, CblasTrans, m_t, m_t, 1, -1.0, q_vec, m_t, p_vec, 1, 1.0, P, m_t);

			for(int i=0;i<m_t;i++) dbl_q_vec[i] = (double)q_vec[i];
			cblas_dgemv(CblasRowMajor,CblasNoTrans, m_t, m_t, dbl_err, K_inv, m_t, dbl_q_vec, 1, 1.0, alpha, 1); //alpha updates
		*/	
		
		}; //delta condition end
	
	}; //init condition end
	
	
	
	
	
	m_out[0] = m[0];
	for(int i=0; i<m[0]; i++) alpha_out[i] = alpha[i];
	for(int i=0; i<m[0]; i++) {
		for(int j=0; j<NIN; j++) dict_out[i*NIN+j] = dict[i*NIN+j];
	};
	
}; 
