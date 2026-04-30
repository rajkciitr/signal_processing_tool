"""
Created on Tue Jun 11 17:29:18 2024
"""
import numpy as np
from matplotlib import pyplot as plt
import librosa

sig, rate = librosa.load('E:/database/AIIMS/phonation/CM04.wav', sr=16000)
#sig = sig[:8000]
min_pitch = 75
max_pitch = 450
max_num_cands = 9    # make sure  max_num_cands > max_pitch / min_pitch
silence_thres = .03  # within [ 0, 1 ]
voicing_thres = .45  # within [ 0, 1 ]
octave_cost = .01    # within [ 0, 1 ]
octave_jump_cost = .35
voiced_unvoiced_cost = .14  # within [ 0, 1 ]
nFFT = 512
inf = 99.0

initial_len = len( sig )


def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def indexes(y,sil_val,time_arr):
    dy = y[1:]-y[:-1] #np.diff(y)
    peaks = np.zeros(max_num_cands-1,dtype=int)
    rel_val=np.ones(max_num_cands)*sil_val
    max_pos=np.ones(max_num_cands)*inf
    k=max_num_cands-2
    
    for i in range(1,len(y)-1):
        if((dy[i] < 0.0) & (dy[i-1] > 0.0) & (y[i] > 0.5 * voicing_thres)):
            peaks[k]=i
            rel_val[k]= y[i] - octave_cost * np.log2( time_arr[i] * min_pitch ) 
            max_pos[k] = time_arr[i]
                       
            k=k-1
            if(k<0):
                break
    

    if(k>1):
         
        start_id = max_num_cands-1-k
        end_id = max_num_cands-2
        for mi in range(start_id,end_id):
            for ni in range(start_id,(start_id + end_id - mi)):
                if (rel_val[ni] > rel_val[ni+1]):
                    temp = rel_val[ni];
                    rel_val[ni] = rel_val[ni+1];
                    rel_val[ni+1] = temp;
                    temp = max_pos[ni];
                    max_pos[ni] = max_pos[ni+1];
                    max_pos[ni+1] = temp;

    return rel_val, max_pos

total_time = initial_len / float( rate )
domain = np.linspace( 0, total_time, initial_len )

max_place_poss  = 1.0 / min_pitch
min_place_poss  = 1.0 / max_pitch
#to silence formants
min_place_poss2 = 0.5 / max_pitch
pds_per_window = 2.4

#degree of oversampling is 4
time_step = 0.008 #( pds_per_window / 4.0 ) / min_pitch
w_len = 0.032  #pds_per_window / min_pitch
#correcting for time_step
octave_jump_cost     *= .01 / time_step
voiced_unvoiced_cost *= .01 / time_step

#finding number of samples per frame and time_step
frame_len = 512 #int( w_len * rate + .5 )
time_len  = 160 #int( time_step  * rate + .5 )

#finding the global peak the way Praat does
global_peak = max( abs( sig - sig.mean() ) )
window = np.hanning( frame_len )
x_fft = np.fft.fft( window )
r_w = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
r_w = r_w[ : int( frame_len / pds_per_window ) ]

#creating an array of the points in time corresponding to sampled
#autocorrelation of the signal ( r_x )
time_array = np.linspace( 0 , w_len / pds_per_window, int( frame_len / pds_per_window )  )
frame_num = int((initial_len-frame_len)/time_len)+1
#initializing list of candidates for F_0, and their strengths
best_cands = np.zeros((frame_num,max_num_cands))
strengths = np.zeros((frame_num,max_num_cands))

start_i = 0
for fr in range(frame_num): # start_i < initial_len - frame_len :
    end_i = start_i + frame_len
    segment = sig[ start_i : end_i ]
    start_i += time_len

    local_mean = segment.mean()
    segment = segment - local_mean
    segment *= window
    local_peak = max( abs( segment ) )
    
    #calculating autocorrelation, based off steps 3.2-3.10
    intensity = local_peak / float( global_peak )
    sil_strengh = voicing_thres + max( 0, 2 - ( intensity /
            ( silence_thres / ( 1 + voicing_thres ) ) ) )
    
    x_fft = np.fft.fft( segment )
    r_a = np.real( np.fft.fft( x_fft * np.conjugate( x_fft ) ) )
    r_a = r_a[ : int( frame_len / pds_per_window ) ]

    r_x = r_a / r_w
    r_x /= r_x[ 0 ]

    rel_val,max_places=indexes(r_x,sil_strengh,time_array)
    
    best_cands[fr,:] = max_places 
    strengths[fr,:] = rel_val

#Calculates smallest costing path through list of candidates ( forwards ),
#and returns path.
best_total_cost, best_total_path = -inf, []
#for each initial candidate find the path of least cost, then of those
#paths, choose the one with the least cost.
for cand in range( max_num_cands):
    start_val = best_cands[ 0 ][ cand ]
    total_path = [ start_val ]
    level = 1
    prev_delta = strengths[ 0 ][ cand ]
    maximum = -inf
    while level < frame_num :
        prev_val = total_path[ -1 ]
        best_val  = inf
        for j in range( len( best_cands[ level ] ) ):
            cur_val   = best_cands[ level ][ j ]
            cur_delta =  strengths[ level ][ j ]
            cost = 0
            cur_unvoiced  = cur_val  == inf or cur_val  < min_place_poss2
            prev_unvoiced = prev_val == inf or prev_val < min_place_poss2

            if cur_unvoiced:
                #both voiceless
                if prev_unvoiced:
                    cost = 0
                #voiced-to-unvoiced transition
                else:
                    cost = voiced_unvoiced_cost
            else:
                #unvoiced-to-voiced transition
                if prev_unvoiced:
                    cost = voiced_unvoiced_cost
                #both are voiced
                else:
                    cost = octave_jump_cost * abs( np.log2( cur_val /
                                                       prev_val ) )

            #The cost for any given candidate is given by the transition
            #cost, minus the strength of the given candidate
            value = prev_delta - cost + cur_delta
            if value > maximum: maximum, best_val = value, cur_val

        prev_delta = maximum
        total_path.append( best_val )
        level += 1

    if maximum > best_total_cost:
        best_total_cost, best_total_path = maximum, total_path

f_0 = np.array( best_total_path )

median_F_0 = np.zeros(len( f_0))
for i in range( len( f_0 ) ):
    #if f_0 is voiceless assign occurance of peak to inf -> when divided
    #by one this will give us a frequency of 0, corresponding to a unvoiced
    #frame
    if f_0[ i ] < max_place_poss and f_0[ i ] > min_place_poss :
        median_F_0[i] = 1/f_0[i]
    

#%%
print(np.mean(median_F_0))
print('act=',108.4509)

plt.subplot( 211 )
plt.plot( domain, sig )
plt.title( "Synthesized Signal" )
plt.ylabel( "Amplitude" )
plt.subplot( 212 )
plt.plot( np.linspace( 0, total_time, len( median_F_0 ) ), median_F_0 )
plt.ylim([0,450])
plt.title( "Frequencies of Signal" )
plt.xlabel( "Samples" )
plt.ylabel( "Frequency" )
plt.suptitle( "Comparison of Synthesized Signal and it's Calculated Frequencies" )
plt.show()

# aaa=np.reshape(time_array,[1,213])
# np.savetxt('time.csv',aaa,fmt='%f',delimiter=',')