"""
Created on 7/21/22

@author: qinyuzhu
"""

import numpy as np

import scipy . stats
import matplotlib . pyplot as plt

"""
frequency domain feature extraction of mechanical vibration signals
"""

class  Fea_Extra ( ) :
    def  __init__ ( self , Signal , Fs =  25600 ) :
        self . signal = Signal
        self . Fs = Fs

    def  Time_fea ( self , signal_ ) :
        """
        Extract 11 types of time domain features
        """
        N =  len ( signal_ )
        y = signal_
        t_mean_1 = np . mean ( y )                                     # 1_mean (average amplitude)

        t_std_2   = np . std ( y , ddof = 1 )                              # 2_standard deviation

        t_fgf_3   =  ( ( np . mean ( np . sqrt ( np . abs (y) ) ) ) )** 2            # 3_square root amplitude

        t_rms_4   = np . sqrt ( ( np . mean ( y ** 2 ) ) )                       # 4_RMS rms

        t_pp_5    =  0.5 * ( np . max (y) - np . min (y) )                      # 5_ peak (refer to Dr. Zhou Hongti senior sister apprentice thesis)

        #t_skew_6 = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
        t_skew_6    = scipy . stats . skew ( y )                          # 6_skewness

        #t_kur_7 = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
        t_kur_7 = scipy . stats . kurtosis ( y )                         # 7_kurtosis

        t_cres_8   = np . max ( np . abs ( y ) ) / t_rms_4                     # 8_ Crest Factor

        t_clear_9   = np . max ( np . abs ( y ) ) / t_fgf_3 # 9_clearance                    factor

        t_shape_10 =  ( N * t_rms_4 ) /( np . sum ( np . abs ( y ) ) )            # 10_Shape fator

        t_imp_11   =  ( np . max ( np . abs ( y ) ) ) /( np . mean ( np . abs ( y ) ) )   # 11_ Impulse Fator

        t_fea = np . array ( [ t_mean_1 , t_std_2 , t_fgf_3 , t_rms_4 , t_pp_5 ,
                          t_skew_6 ,    t_kur_7 ,   t_cres_8 ,   t_clear_9 , t_shape_10 , t_imp_11 ] )

        #print("t_fea:",t_fea.shape,'\n', t_fea)
        return t_fea

    def  Fre_fea ( self , signal_ ) :
        """
        Extract 13 types of frequency domain features
        :param signal_:
        :return:
        """
        L =  len ( signal_ )
        PL =  abs ( np. FFT . FFT ( signal_/ L ) ) [ :  int ( L / 2 ) ]
        PL [ 0 ]  =  0
        F = np. FFT . Fftfreq ( L ,.1  / self . Fs ) [:  int ( L / 2 ) ]
        x = F
        y = PL
        K =  len ( y )
        # print("signal_.shape:",signal_.shape)
        # print("PL.shape:", PL.shape)
        # print("L:", L)
        # print("K: ", K)
        # print("x:",x)
        # print("y:",y)

        f_12 = np . mean ( y )

        f_13 = np . var ( y )

        f_14 =  ( np . sum ( ( y - f_12 ) ** 3 ) ) /( K *  ( ( np . sqrt ( f_13 ) ) ** 3 ) )

        f_15 =  ( np . sum ( ( y - f_12 ) ** 4 ) ) /( K *  ( ( f_13 ) ** 2 ) )

        f_16 =  ( np . sum ( x * y ) ) /( np . sum ( y ) )

        f_17 = np . sqrt ( ( np . mean ( ( ( x - f_16 ) ** 2 ) * ( y ) ) ) )

        f_18 = np . sqrt ( ( np . sum ( ( x ** 2 ) * y ) ) /( np . sum ( y ) ) )

        f_19 = np . sqrt ( ( np . sum ( ( x ** 4 ) * y ) ) /( np . sum ( ( x ** 2 ) * y ) ) )

        f_20 =  ( np . sum ( ( x ** 2 ) * y ) ) /( np . sqrt ( ( np . sum ( y ) ) * ( np . sum ( ( x ** 4 ) * y ) ) ) )

        f_21 = f_17 / f_16

        f_22 =  ( np . sum ( ( ( x - f_16 ) ** 3 ) * y ) ) /( K *  ( f_17 ** 3 ) )

        f_23 =  ( np . sum ( ( ( x - f_16 ) ** 4 ) * y ) ) /( K *  ( f_17 ** 4 ) )

        #f_24 = (np.sum((np.sqrt(x-f_16))*y))/(K * np.sqrt(f_17)) # There is a minus sign under the root sign of f_24, so remove it first


        #print("f_16:",f_16)

        #f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24])
        f_fea = np . array ( [ f_12 , f_13 , f_14 , f_15 , f_16 , f_17 , f_18 , f_19 , f_20 , f_21 , f_22 , f_23 ] )

        #print("f_fea:",f_fea.shape,'\n', f_fea)
        return f_fea

    def  Both_Fea ( self ) :
        """
        :return: time domain, frequency domain feature array
        """
        t_fea = self . Time_fea ( self . signal )
        f_fea = self . Fre_fea ( self . signal )
        fea = np . append ( np . array ( t_fea ) , np . array ( f_fea ) )
        #print("fea:" , fea.shape,'\n', fea)
        return fea



if __name__ == '__main__':
    pass
