# dipet_mixed_krls - mixed_krls v0.1 source codes <br>

# Requirement: MKL library <br>
1. Please install a MKL library in /opt/intel/ <br>
2. Type: source /opt/intel/mkl/bin/mklvars.sh intel64 <br>
3. Go to ./blas_ver directory. Non-blas version is not available at this moment. <br>
4. Type: make <br> 

learn_curve_krls.cpp file simulates the learning accuracy of three different arithmetic KRLS.<br>

5. The execution command lines are:<br>
./learn_krls_curve arithmetic_type appl_type  min_#_samples max_#_samples input_length kernel_width ald_param<br>
e.g., ./learn_krls_curve mixed mg30 1 10000 6 0.5 0.001<br>
or <br>
./learn_krls_curve double sunspot 120 2880 3 95 0.0001 (IN THIS application, traning data will have a stride of 120, should be fixed for input length to 3)<br>
<br>
Run<br>
./learn_krls_curve prec_type(double or mixed or single) application_type(sinc_lin_tri or mg30 or sunspot) first_#_samples(5) last_#_samples(50000) feature_size(3)<br> kernel_width(4.0) ald_threshold(0.001)<br>
=><br>
   #_tr_samples    log10(RMSE)  training_time   testing_time         D size<br>
              5       -0.34563     0.00619197     0.00875616              5<br>
             50      -0.611389     0.00492191       0.012069             42<br>
            500       -1.34673      0.0445101       0.012876             65<br>
           5000       -1.83638        0.21237       0.012913             66<br>
          50000        -2.1564        5.63238      0.0156641             68<br>
<br>
NOTE: Do not put a fractional number as an argument.<br> 
