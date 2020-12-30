#Projekt do předmětu ISS (Signály a systémy		#
#Vypracoval: Ondřej Pavlacký -(xpavla15)		#
#Poslední úprava: 16.12.2019 					#
#################################################

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython
import statistics as st
import os
from scipy import stats
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
#Import všemožných užitečných knihoven
#############################################################

#Frekvence
fs = 16000

#Výpočet pro Querry a matice pro násobení
qs1, qfs1 = sf.read('../queries/q1.wav')
qs2, qfs2 = sf.read('../queries/q2.wav')
	
#3-----------------------------------------------------
#a) Zaznačit střední hodnotu
qs1 = qs1 - st.mean(qs1)
qs2 = qs2 - st.mean(qs2)

#b) Signál rozdělíme na segmenty (rámce), posun mezi rámci je 10ms
wlen = 25e-3 * fs
wshift = 15e-3 * fs
woverlap = wlen - wshift
#c) Hammingovo okno 
win = np.hamming(wlen)

#d) Vybraný rámec se doplní nulami, délka N
#511 a ne 512 kvůli indexaci
N = 511 

#e) N/2
N_2= N/2

#Vytvoření matice nul
A = np.zeros( (16, 256) )

#Přidání jedniček po 16ti do každého sloupce
for i in range (0,16): 
	A[i][16*i] = 1
	A[i][16*i+1] = 1
	A[i][16*i+2] = 1
	A[i][16*i+3] = 1
	A[i][16*i+4] = 1
	A[i][16*i+5] = 1
	A[i][16*i+6] = 1
	A[i][16*i+7] = 1
	A[i][16*i+8] = 1
	A[i][16*i+9] = 1
	A[i][16*i+10] = 1
	A[i][16*i+11] = 1
	A[i][16*i+12] = 1
	A[i][16*i+13] = 1
	A[i][16*i+14] = 1
	A[i][16*i+15] = 1

	#Výpočet Q1(query)
	qf1, qt1, q_sgr1 = spectrogram(qs1, qfs1, win, wlen, wshift, N)
	F_q1_sgr_log = 10 * np.log10(abs((q_sgr1+1e-20)**2)) 
	F_q1 = np.matmul(A, q_sgr1)

	#Výpočet Q2(query)
	qf2, qt2, q_sgr2 = spectrogram(qs2, qfs2, win, wlen, wshift, N)
	F_q2_sgr_log = 10 * np.log10(abs((q_sgr2+1e-20)**2)) 
	F_q2 = np.matmul(A, q_sgr2)

############################################################################
#Cyklus na iteraci přes všechny soubory
for j in os.listdir('../sentences'):
	#1-----------------------------------------------------

	#Zvukové soubory byly nahrány ve formátu WAV, Fs = 16000Hz, mono, 16bitů jedne vzorek
	#Import všech zvukových souborů postupně
	s, fs = sf.read('../sentences/' + j)

	#######################################################

	#3-----------------------------------------------------

	#a) Zaznačit střední hodnotu
	s = s - st.mean(s)

	#b) Signál rozdělíme na segmenty (rámce), posun mezi rámci je 10ms
	wlen = 25e-3 * fs
	wshift = 15e-3 * fs
	woverlap = wlen - wshift

	#c) Hammingovo okno 
	win = np.hamming(wlen)

	#d) Vybraný rámec se doplní nulami, délka N
	#511 a ne 512 kvůli indexaci
	N = 511 

	#e) N/2
	N_2= N/2

	#Kód od Kateřiny Žmolíkové, velmi děkuji :)

	#Nahrávku budeme používat celou
	s = s[:s.size]
	#Zjistíme si čas nahráky pro plot
	t = np.arange(s.size) / fs

	#Vykreslení signálu
	
	#Nastavení zobrazení vykreslování
	fig, axs = plt.subplots(3, 1, constrained_layout=True)

	fig.set_size_inches(15, 10)
	fig.suptitle('Nahrávka ' + j, fontsize=18)

	axs[0].margins(0)
	axs[0].plot(t, s)
	axs[0].set(xlabel='Time', ylabel='Signal')
	
	f, t, sgr = spectrogram(s, fs, win, wlen, wshift, N)

	# prevod na PSD
	# (ve spektrogramu se obcas objevuji nuly, kwin, wlen, woverlap, 512)
	# prevod na PSD
	# (ve spektrogramu se obcas objevuji nuly, ktere se nelibi logaritmu, proto +1e-20)

	#f) logaritmické výkonové spektrum
	sgr_log = 10 * np.log10(abs((sgr+1e-20)**2)) 

	#g) Uložení a zobrazí berevného spetrogramu, pokud by jste si jej chtěli vykreslit
	#plt.figure(figsize=(9,3))
	#plt.title('Úloha číslo 3')
	#plt.pcolormesh(t,f,sgr_log)
	#plt.gca().set_title('Název nahrávky: ' + j)
	#plt.gca().set_xlabel('Čas [s]')
	#plt.gca().set_ylabel('Frekvence [Hz]')
	#cbar = plt.colorbar()
	#cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
	#plt.tight_layout()

	#Pokud by jsme si chtěli vykreslit spektrogram dané nahrávky
	#plt.show()

	#################################################################################
	#4-------------------------------------------------------------------------------

	#Matici jedniček a nul jsem si vytvořil předem, není potřeba ji tvořit vícekrát
	#Nahrání Query a jeho výpočet taky stačí udělat jednou

	# F = A*P(sgr)
	F = np.matmul(A, sgr)

	#Výpočet logaritmu pro vykreslení
	F_log = 10 * np.log10(abs((F+1e-20)**2)) 

	F = F.transpose()
	#Vykreslení grafu features

	axs[1].pcolormesh(F_log)
	array= [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
	axs[1].set_xticks(np.arange(0,len(F_log[0]),50), minor=False)
	axs[1].set_xticklabels(array, fontdict=None, minor=False)
	
	axs[1].set(ylabel='Features', xlabel='Time')


	###################################################################################
	#5---------------------------------------------------------------------------------	

	#Transponování matic pro lepší výpočet
	F_transpose = F
	F_q1_transpose = F_q1.transpose()
	F_q2_transpose = F_q2.transpose()

	#Výpočet počtu iterací, abych nepřeskočil délku věty, potom by query bylo nedefinované
	pocet_testu_q1 = len(F_transpose)- len(F_q1_transpose)
	pocet_testu_q2 = len(F_transpose)- len(F_q2_transpose)

	#Vytvoření polí pro nahrávání hodnot
	result_q1= []
	result_q2= []

	#Cyklus pro výpočet Pearsonova korelačního exponentu pro q1
	#pocet_testu_q1 = počet iterací přes celou větu
	for i in range(0,len(F_transpose)):
		vysledek_q1 = 0
		
		if i < pocet_testu_q1:
			
		#Počet iterací přes celou délku query
			for c in range(0,len(F_q1_transpose)):
				koef_q1, useless = stats.pearsonr(F_transpose[i+c], F_q1_transpose[c])
				vysledek_q1+=koef_q1
		#Přidávám  do pole výslednou hodnotu
			result_q1= np.append(result_q1,vysledek_q1)
			result_q1= abs(result_q1)
	#Tady počítám prvek prvek kde se dostanu nad vybranou hodnotu score
	for i in range (0,len(result_q1)):
		if(result_q1[i]) > 0.8*len(F_q1_transpose):
			#Výpis výskytu query, pod graf skore
			axs[2].text(0.5,-0.1, "První výskyt q1 : " + str(i*160) + " Konec výskytu: " + str(i*160+len(qs1)), size=12, ha="left", va="top")
			F_q1_neco= s[i:i+len(qs1)]
			sf.write('../hits/q1_hit_si761.wav', F_q1_neco, 16000)
			break


	#Cyklus pro výpočet Pearsonova korelačního exponentu pro q2
	#pocet_testu_q2 = počet iterací přes celou větu
	for i in range(0,len(F_transpose)):
		vysledek_q2 = 0
		if i <= pocet_testu_q2:
			
		#Počet iterací přes celou délku query
			for c in range(0,len(F_q2_transpose)):
				koef_q2, useless = stats.pearsonr(F_transpose[i+c], F_q2_transpose[c])
				vysledek_q2+=koef_q2
		#Přidávám  do pole výslednou hodnotu
			result_q2= np.append(result_q2,vysledek_q2)
			result_q2= abs(result_q2)
	#Tady počítám prvek prvek kde se dostanu nad vybranou hodnotu score
	for i in range (0,len(result_q2)):
		if(result_q2[i]) > 0.8*len(F_q2_transpose):
			#Výpis výskytu query, pod graf score
			axs[2].text(0.5,-0.1, "První výskyt q2 : " + str(i*160) + " Konec výskytu: " + str(i*160+len(qs2)), size=12, ha="left", va="top")
			F_q2_neco= s[i:i+len(qs2)]
			sf.write('../hits/q2_hit_sx41.wav', F_q2_neco, 16000)
			break


	#Vykreslení grafu
	#Nastavení legendy ke křivce skóre
	prvni = axs[2].plot(np.arange(len(result_q1)) /100  , result_q1/len(F_q1_transpose), color='b', label='Paperweight -q1')
	druhy = axs[2].plot(np.arange(len(result_q2)) /100 , result_q2/len(F_q2_transpose), color='r', label='Etiquette -q2')
	plt.ylim(0, 1)
	plt.xlim(0, len(F_transpose)/100)
	axs[2].set(xlabel='Time', ylabel='Score')
	axs[2].margins(0.0, x=None, y=1, tight=None)

	#Vypsání legendy k třetímu grafu
	plt.legend()

	#Celkové vykreslení všech grafů s osami a popisy
	plt.show()

##################################################################################