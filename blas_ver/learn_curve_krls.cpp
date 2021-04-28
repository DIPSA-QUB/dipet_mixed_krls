#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <string.h>
//#define NIN 3 			//Problem B NIN, MAXDICT should be defined before xx_krls.h
//#define NIN 2 //Problem A
//#define NIN 7 //MG
#define MAXDICT 2000
#include "single_krls.h"
#include "double_krls.h"
#include "mixed_krls.h" //mixed_krls.h should be defined behind "double_krls.h"
#include "get_timing.h"
//#define NU 0.002
//#define BETA 3.4
#define REGUL 0.0000

using namespace std;

//#define N_TR_SAMP 10000
//#define N_TST_SAMP 1000
//double x[NIN], x_tst[NIN], TRAINX[NIN*N_TR_SAMP], TRAINY[N_TR_SAMP], TSTX[NIN*N_TST_SAMP], TSTY[N_TST_SAMP];

int main(int argc, char* argv[]) {
	
	int mode, appl_mode;
	double BETA, NU;
	//double *x, *x_tst, *TRAINX, *TRAINY, *TSTX, *TSTY;

	int max_nsample, min_nsample, N_TRIAL, NIN, N_TST_SAMP, N_TR_SAMP;
	char *arith_type, *appl_type;
	const char *mixed, *dble, *sgle, *sinc_lin, *sinc_lin_tri, *mg30, *sunspot;
	
	if(argc!=8) {fprintf(stderr,"e.g., ./learn_curve_krls arith_type appl_type min_#_tr_samples max_#_tr_samples input_length beta nu\n"); return 1;}
	
	mixed = "mixed";
	dble = "double";
	sgle = "single";
	sinc_lin = "sinc_lin";
	sinc_lin_tri = "sinc_lin_tri";
	mg30 = "mg30";
	sunspot = "sunspot";
	
	arith_type = argv[1]; appl_type = argv[2]; min_nsample = atoi(argv[3]); max_nsample = atoi(argv[4]); NIN = atof(argv[5]); BETA = atof(argv[6]); NU=atof(argv[7]);
	if(!strcmp(sgle,arith_type)) {mode = 1;} //single
	else if(!strcmp(dble,arith_type)) {mode = 2;} //double
	else if(!strcmp(mixed,arith_type)) {mode = 3;} //mixed
	else {fprintf(stderr,"your refinement type is not defined."); return -1;}
	
	if(!strcmp(sinc_lin,appl_type)) {appl_mode = 1; N_TR_SAMP=10000; N_TST_SAMP=1000;} //sinc_linear application
	else if (!strcmp(sinc_lin_tri,appl_type)) {appl_mode = 2; N_TR_SAMP=10000; N_TST_SAMP=1000;} //sinc_linear_trigonometric application
	else if (!strcmp(mg30,appl_type)) {appl_mode = 3; N_TR_SAMP=10000; N_TST_SAMP=1000;} //mackey glass 30 applications
	else if (!strcmp(sunspot,appl_type)) {appl_mode = 4; N_TR_SAMP=2880; N_TST_SAMP=335;} //sunspot application
	else
	{printf("\nNo defined problem\n"); return -1;}
	
	ifstream xin, yin, xin_tst, yin_tst;
	int init[1] ;
	char linex[10], liney[10];
	int m_out[1];

		
	float* sdict_out = (float*) malloc(NIN*MAXDICT*sizeof(float));
	float* salpha_out = (float*) malloc(MAXDICT*sizeof(float));
	float sx[NIN], sx_tst[NIN];
	float spred[1], strain_err[1];
	float sy, sy_tst;
	double* dict_out = (double*) malloc(NIN*MAXDICT*sizeof(double));
	double* alpha_out = (double*) malloc(MAXDICT*sizeof(double));
	double* x = (double* )malloc(NIN*sizeof(double));
	double* x_tst = (double* )malloc(NIN*sizeof(double)); 
	double* TRAINX = (double* )malloc(NIN*N_TR_SAMP*sizeof(double));
	double* TRAINY = (double* )malloc(N_TR_SAMP*sizeof(double)); 
	double* TSTX = (double* )malloc(NIN*N_TST_SAMP*sizeof(double)); 
	double* TSTY= (double* )malloc(N_TST_SAMP*sizeof(double));
	double pred[1], train_err[1];
	double y, err, y_tst, tmp_err;
	double btime_tr, etime_tr, btime_tst, etime_tst, btime_krls, etime_krls, tot_internal_krls_time;
	
	tot_internal_krls_time = 0.0;
	
	if(appl_mode==1) {
	xin.open("../data/ProblemA_2D/X_train.dat"); if(!xin) {cout<<"Can't read X_train"<<endl;};
	yin.open("../data/ProblemA_2D/Y_train.dat"); if(!yin) {cout<<"Can't read Y_train"<<endl;};
	xin_tst.open("../data/ProblemA_2D/X_tst.dat"); if(!xin_tst) {cout<<"Can't read X_tst"<<endl;};
	yin_tst.open("../data/ProblemA_2D/Y_tst.dat"); if(!yin_tst) {cout<<"Can't read Y_tst"<<endl;};
	}
	else if(appl_mode==2){
	xin.open("../data/ProblemB_3D/X_train.dat"); if(!xin) {cout<<"Can't read X_train"<<endl;};
	yin.open("../data/ProblemB_3D/Y_train.dat"); if(!yin) {cout<<"Can't read Y_train"<<endl;};
	xin_tst.open("../data/ProblemB_3D/X_tst.dat"); if(!xin_tst) {cout<<"Can't read X_tst"<<endl;};
	yin_tst.open("../data/ProblemB_3D/Y_tst.dat"); if(!yin_tst) {cout<<"Can't read Y_tst"<<endl;};
	}
        else if(appl_mode==3){
        xin.open("../data/MG30/X_train.dat"); if(!xin) {cout<<"Can't read X_train"<<endl;};
        yin.open("../data/MG30/Y_train.dat"); if(!yin) {cout<<"Can't read Y_train"<<endl;};
        xin_tst.open("../data/MG30/X_tst.dat"); if(!xin_tst) {cout<<"Can't read X_tst"<<endl;};
        yin_tst.open("../data/MG30/Y_tst.dat"); if(!yin_tst) {cout<<"Can't read Y_tst"<<endl;};
        }
        else if(appl_mode==4){
        xin.open("../data/Sunspot/X_train.dat"); if(!xin) {cout<<"Can't read X_train"<<endl;};
        yin.open("../data/Sunspot/Y_train.dat"); if(!yin) {cout<<"Can't read Y_train"<<endl;};
        xin_tst.open("../data/Sunspot/X_tst.dat"); if(!xin_tst) {cout<<"Can't read X_tst"<<endl;};
        yin_tst.open("../data/Sunspot/Y_tst.dat"); if(!yin_tst) {cout<<"Can't read Y_tst"<<endl;};
        }
	else
	{printf("\nNo defined problem\n"); return -1;}
	
	
	for(int i=0; i<N_TR_SAMP; i++){
		for(int j=0; j<NIN; j++) { xin >> linex; TRAINX[i*NIN+j] = strtod(linex, NULL);}// X test inputs
		yin >> liney; TRAINY[i] = strtod(liney, NULL);
	};
	
	for(int i=0; i<N_TST_SAMP; i++){
		for(int j=0; j<NIN; j++) { xin_tst >> linex; TSTX[i*NIN+j] = strtod(linex, NULL);}// X test inputs
		yin_tst >> liney; TSTY[i] = strtod(liney, NULL);
	};

	int train_num = min_nsample; N_TRIAL = log10(max_nsample/min_nsample)+1;
	if(appl_mode==4) {N_TRIAL = 24; }
	double fl_test_num=N_TST_SAMP;
	
	cout <<setw(15)<< "#_tr_samples"  <<setw(15) << "log10(RMSE)" << setw(15) << "training_time" << setw(15)<< "testing_time" << setw(15)<<"D size"<<setw(15)<<"internal training time"<<endl;
	
	for (int trials= 0; trials < N_TRIAL ; trials++) {
		init[0] = 1;
		//TRAINING
		btime_tr = get_cur_time();
		for (int i=0; i<train_num; i++) {
			for(int j=0; j<NIN; j++) {
				x[j] = TRAINX[(i)*NIN+j];
				if(mode==1) sx[j] = (float)x[j];
			}// X test inputs
			
			y = TRAINY[i]; if(mode==1) sy = (float)y;
			
	//	btime_krls = get_cur_time();
			if(mode==1)
				single_krls(BETA, NU,REGUL, sx, sy, sdict_out, salpha_out, m_out, strain_err, init, NIN);
			else if(mode==2)
				double_krls(BETA, NU,REGUL, x, y, dict_out, alpha_out, m_out, train_err, init, NIN);
			else if(mode==3) {
	//	btime_krls = get_cur_time();
				mixed_krls(BETA, NU,REGUL, x, y, dict_out, alpha_out, m_out, train_err, init, NIN);
	//	etime_krls = get_cur_time();
			}
			else {printf("your refinement type is not defined."); return -1;}
	//	etime_krls = get_cur_time();
	//	tot_internal_krls_time += etime_krls - btime_krls;
		}
		etime_tr = get_cur_time();		
		
		//TESTING
		tmp_err = 0;
		btime_tst = get_cur_time();
		for (int i=0; i<N_TST_SAMP; i++) {
			
			for(int j=0; j<NIN; j++) {x_tst[j] = TSTX[i*NIN+j]; if(mode==1) sx_tst[j] = (float)x_tst[j];}// X test inputs
			y_tst = TSTY[i]; if(mode==1) sy_tst = (float) y_tst;	
			
			if(mode==1)
				sinference_krls(sx_tst, sdict_out, salpha_out, BETA, m_out[0], spred, NIN) ;
			else
				inference_krls(x_tst, dict_out, alpha_out, BETA, m_out[0], pred, NIN) ;
			
			if(mode==1) pred[0] = (double)spred[0];
			err = y_tst - pred[0];
			tmp_err = tmp_err + err*err;
		}
		etime_tst = get_cur_time();	
	
	cout<<setw(15)<<train_num<<setw(15)<<log10(sqrt(tmp_err/fl_test_num))<< setw(15)<< etime_tr-btime_tr << setw(15)<< etime_tst-btime_tst<< setw(15)<<m_out[0]<<setw(15) << tot_internal_krls_time<<endl;
	
	if(appl_mode==4) {train_num += min_nsample; }
	else {train_num = train_num*10;}
	}
	
	return 0;

}
