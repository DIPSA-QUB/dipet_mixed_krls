#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

#define CHK 0
#define MIXED 1
#define PRECTYPE double
#define NIN 10
#define MAXDICT 1000
#define NU 0.00001
#define BETA 4.0
#define REGUL 0.0000
#include <mixed_krls.h>

using namespace std;

#define N_TR_SAMP 4000
#define N_TST_SAMP 900

#define N_TRIAL 4

int main( )
{
	ifstream xin, yin;
	ifstream xin_tst, yin_tst;
	int init[1] ;
	char linex[10], liney[10];
	int m_out[1];
	PRECTYPE* dict_out = (PRECTYPE*) malloc(NIN*MAXDICT*sizeof(PRECTYPE));
	PRECTYPE* alpha_out = (PRECTYPE*) malloc(MAXDICT*sizeof(PRECTYPE));

    PRECTYPE x[NIN], y;
     
      PRECTYPE x_tst[NIN], y_tst;
     
     PRECTYPE TRAINX[NIN*N_TR_SAMP];
     PRECTYPE TRAINY[N_TR_SAMP];
     
     PRECTYPE TSTX[NIN*N_TST_SAMP];
     PRECTYPE TSTY[N_TST_SAMP];

	PRECTYPE pred[1]; PRECTYPE err; 
	PRECTYPE train_err[1];
	//double train_err[1];
double btime_tr, etime_tr, btime_tst, etime_tst;


	PRECTYPE tmp_err = 0;
	

	
		xin.open("mX_train.dat"); if(!xin) {cout<<"Can't read X_train"<<endl;};
		yin.open("mY_train.dat"); if(!yin) {cout<<"Can't read Y_train"<<endl;};
		xin_tst.open("mX_tst.dat"); if(!xin_tst) {cout<<"Can't read X_tst"<<endl;};
		yin_tst.open("mY_tst.dat"); if(!yin_tst) {cout<<"Can't read Y_tst"<<endl;};
		
		init[0] = 1;

		for(int i=0; i<N_TR_SAMP; i++){
			
			for(int j=0; j<NIN; j++) { xin >> linex; TRAINX[i*NIN+j] = strtod(linex, NULL);}// X test inputs
			yin >> liney; TRAINY[i] = strtod(liney, NULL);
			
		};
		
		for(int i=0; i<N_TST_SAMP; i++){
			
			for(int j=0; j<NIN; j++) { xin_tst >> linex; TSTX[i*NIN+j] = strtod(linex, NULL);}// X test inputs
			yin_tst >> liney; TSTY[i] = strtod(liney, NULL);
			
		};

		//cout<<TRAINX[2]<<endl;


int train_num = 4;
PRECTYPE fl_test_num; fl_test_num=N_TST_SAMP;


		for (int trials= 0; trials < N_TRIAL ; trials++) {
		//TRAINING
		btime_tr = get_cur_time();
			for (int i=0; i<train_num; i++) {
					for(int j=0; j<NIN; j++) {x[j] = TRAINX[(i)*NIN+j];}// X test inputs
					y = TRAINY[i];
					
					train_krls(BETA, NU,REGUL, x, y, dict_out, alpha_out, m_out, train_err, init);
				}
		etime_tr = get_cur_time();		
		//TESTING
			tmp_err = 0;
			init[0] = 1;
			
		btime_tst = get_cur_time();
			for (int i=0; i<N_TST_SAMP; i++) {
		
		
				for(int j=0; j<NIN; j++) {x_tst[j] = TSTX[i*NIN+j];}// X test inputs
				y_tst = TSTY[i];	
				//cout<<y_tst<<endl;
		inference_krls(x_tst, dict_out, alpha_out, BETA, m_out[0], pred) ;
				//cout<<pred[0]<<endl;
				err = y_tst - pred[0];
				tmp_err = tmp_err + err*err;
		
				}
		etime_tst = get_cur_time();	
			
			cout<<log10(sqrt(tmp_err/fl_test_num))<<"  "<< etime_tr-btime_tr <<" "<< etime_tst-btime_tst<<"  "<<m_out[0]<<endl;
			//cout<<pred[0]<<endl;
		train_num = train_num*10;
		}
		


		return 0;

}
