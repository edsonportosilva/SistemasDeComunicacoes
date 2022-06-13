# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Espectro de modulações de onda contínua

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Carrega-sinais-de-áudio" data-toc-modified-id="Carrega-sinais-de-áudio-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Carrega sinais de áudio</a></span><ul class="toc-item"><li><span><a href="#Plota-sinais-no-domínio-do-tempo" data-toc-modified-id="Plota-sinais-no-domínio-do-tempo-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Plota sinais no domínio do tempo</a></span></li><li><span><a href="#Plota-densidades-espectrais-de-potência" data-toc-modified-id="Plota-densidades-espectrais-de-potência-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Plota densidades espectrais de potência</a></span></li></ul></li><li><span><a href="#Caso-1:-AM-DSB-SC" data-toc-modified-id="Caso-1:-AM-DSB-SC-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Caso 1: AM-DSB-SC</a></span></li><li><span><a href="#Caso-2:-AM-SSB" data-toc-modified-id="Caso-2:-AM-SSB-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Caso 2: AM-SSB</a></span></li></ul></div>

# +
from scipy.signal import firwin, lfilter, freqz, hilbert
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import IPython
from scipy.io import wavfile

import IPython

# bandpass FIR filter.
def bandpass_firwin(ntaps, lowcut, highcut, Fs, window='hamming'):    
    taps = firwin(ntaps, [lowcut, highcut], fs=Fs, pass_zero=False, window=window, scale=True)
    return taps

# lowpass FIR filter.
def lowpass_firwin(ntaps, fcut, Fs, window='hamming'):    
    taps = firwin(ntaps, fcut, fs=Fs, window=window, scale=True)
    return taps

# função para calcular a potência de um sinal
def potSinal(x):
    return (x**2).mean()

def filterNoDelay(h, x):
    """
    h: impulse response (symmetric)
    x: input signal 
    y: output signal
    """   
    N = h.size
    x = np.pad(x, (0, int(N/2)),'constant')
    y = lfilter(h,1,x)
    
    return y[int(N/2):y.size]


# +
from IPython.core.display import HTML
from IPython.core.pylabtools import figsize

figsize(10, 3)
# -

HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")

# ## Carrega sinais de áudio

# +
# fa = 48000   # frequência de amostragem do áudio

# carrega arquivo de áudio
fa, m1 = wavfile.read('voz1.wav')
fa, m2 = wavfile.read('voz2.wav')

m1 = m1[:,0]
m2 = m2[:,0]

t = np.arange(0, len(m1))*1/fa
m1 = m1/abs(m1).max()
m2 = m2/abs(m2).max()
# -

# ### Plota sinais no domínio do tempo

# + hide_input=false
plt.figure()
plt.plot(t, m1, linewidth=0.25, label = 'audio signal 1');
plt.xlim(0,np.max(t))
plt.ylim(-1,1)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('$m_1(t)$');

plt.figure()
plt.plot(t, m2, linewidth=0.25, label = 'audio signal 2');
plt.xlim(0,np.max(t))
plt.ylim(-1,1)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('$m_2(t)$');
# -

# ### Plota densidades espectrais de potência

# + hide_input=true
plt.psd(m1, Fs=fa, label ='$m_1(t)$',linewidth=1.5,NFFT=2048, sides='twosided');
plt.psd(m2, Fs=fa, label ='$m_2(t)$',linewidth=1.5,NFFT=2048, sides='twosided')
plt.legend();
plt.xlim(-fa/2, fa/2);
plt.ylim(-110,);
# -

IPython.display.Audio('voz1.wav')

IPython.display.Audio('voz2.wav')

# ## Caso 1: AM-DSB-SC

# +
# Exemplo 1 (AM-DSB-SC):

Fs     = 48e3     # frequência de amostragem do sinal de áudio
fc_tx  = 10e3     # frequência da portadora
B_sig  = 4e3      # largura de banda do sinal de áudio
ntaps  = 4096+1   # número de coeficientes dos filtros
SNR    = 50       # SNR desejada em dB
π      = np.pi
θ      = 0

# frequências de corte do filtro passa-faixa:
lowcut  = fc_tx - B_sig
highcut = fc_tx + B_sig

h = bandpass_firwin(ntaps, lowcut, highcut, Fs)
g = lowpass_firwin(ntaps, B_sig, Fs)

w, H = freqz(h, fs=Fs, worN=4096)
w, G = freqz(g, fs=Fs, worN=4096)

# plota o valor absoluto das respostas em frequência dos filtros
plt.plot(w, 10*np.log10(np.abs(H)), linewidth=2, label = 'H(f)')
plt.plot(w, 10*np.log10(np.abs(G)), linewidth=2, label = 'G(f)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Frequency response of the filters')
plt.legend()
plt.grid(True)
plt.xlim(min(w),max(w));

# +
x = filterNoDelay(g, m1)

t = np.arange(0, len(x))*1/Fs
plt.figure(figsize =(12,4))
plt.plot(t, x, linewidth = 0.2, label='$m_1(t) \\ast g(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')
plt.grid(True)
plt.xlim(min(t),max(t));
plt.legend()

plt.figure(figsize =(12,4))
plt.psd(x, Fs=Fs, label='PSD of $m_1(t) \\ast g(t)$ (filtered signal)', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(m1, Fs=Fs, label='PSD of $m_1(t)$', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(0,Fs/2);
plt.grid(True);
# -

wavfile.write('voz1_filtrada.wav', int(Fs), x.astype(np.float32))

IPython.display.Audio('voz1_filtrada.wav')

# +
# Modula sinal de voz DSB-SC e plota espectro do sinal modulado:
s_dsb = x*np.cos(2*π*fc_tx*t + θ)

Ps = potSinal(s_dsb) # calcula potência do sinal modulado
Pn = Ps/10**(SNR/10) # calcula potência do ruído na banda do sinal
N0 = Pn/(4*B_sig)    # calcula N0

σ2 = N0*Fs # variância 
μ  = 0         # média

# Adiciona ruído branco
ruido    = normal(μ, np.sqrt(σ2), len(s_dsb))
ruido_pf = filterNoDelay(h, ruido)

plt.figure(figsize =(12,4))
plt.psd(ruido, Fs=Fs, label='DEP do ruído branco',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(s_dsb, Fs=Fs, label='DEP do sinal AM-DSB',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(ruido_pf, Fs=Fs, label='DEP do ruído passa-faixa',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# **Calculando a $\mathrm{SNR}$ pré-demodulador**

SNRpre = 10*np.log10(potSinal(s_dsb)/potSinal(ruido_pf))
print('SNRpre = %.2f dB'%SNRpre)

# +
s_dsb_rx = s_dsb + ruido              # ruído aditivo gaussiano
s_dsb_rx = filterNoDelay(h, s_dsb_rx) # filtragem passa-faixa

s_demod  = s_dsb_rx*np.cos(2*π*fc_tx*t) # demodulação síncrona
x_demod  = filterNoDelay(g, s_demod)    # filtragem passa-baixa

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# **Calculando a $\mathrm{SNR}$ pós-demodulador**

# +
ruido_pb  = ruido_pf*np.cos(2*π*fc_tx*t)
ruido_pb  = filterNoDelay(g, ruido_pb) 

s_pb  = s_dsb*np.cos(2*π*fc_tx*t)
s_pb  = filterNoDelay(g, s_pb) 

SNRpos = 10*np.log10(potSinal(s_pb)/potSinal(ruido_pb))
print('SNRpos = %.2f dB'%SNRpos)
# -

x_demod = x_demod/abs(x_demod).max(0)
wavfile.write('voz1_demodAMDSBSC.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz1_demodAMDSBSC.wav')

# ## Caso 2: AM-SSB

# +
SNR = 50

# frequências de corte do filtro passa-faixa:
lowcut  = fc_tx-B_sig
highcut = fc_tx

h = bandpass_firwin(ntaps, lowcut, highcut, Fs)

# Modula sinal de voz SSB e plota epectro do sinal modulado:
s_ssb = 1/np.sqrt(2)*( x*np.cos(2*π*fc_tx*t) + np.imag(hilbert(x))*np.sin(2*π*fc_tx*t) )

Ps = potSinal(s_ssb)
Pn = Ps/10**(SNR/10)
N0 = Pn/(2*B_sig)

σ2 = N0*Fs # variância 
μ  = 0         # média

# Adiciona ruído branco
ruido    = normal(μ, np.sqrt(σ2), len(s_ssb))
ruido_pf = filterNoDelay(h, ruido)

plt.figure(figsize =(12,4))
plt.psd(ruido, Fs=Fs, label='DEP do ruído branco',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(s_ssb, Fs=Fs, label='DEP do sinal AM-SSB',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(ruido_pf, Fs=Fs, label='DEP do ruído passa-faixa',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# **Calculando a $\mathrm{SNR}$ pré-demodulador**

SNRpre = 10*np.log10(potSinal(s_ssb)/potSinal(ruido_pf))
print('SNRpre = %.2f dB'%SNRpre)

# +
s_ssb_rx = s_ssb + ruido
s_ssb_rx = filterNoDelay(h, s_ssb_rx)

s_demod  = s_ssb_rx*np.cos(2*π*fc_tx*t)
x_demod  = filterNoDelay(g, s_demod)

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# **Calculando a $\mathrm{SNR}$ pós-demodulador**

# +
ruido_pb  = ruido_pf*np.cos(2*π*fc_tx*t)
ruido_pb  = filterNoDelay(g, ruido_pb) 

s_pb  = s_ssb*np.cos(2*π*fc_tx*t)
s_pb  = filterNoDelay(g, s_pb) 

SNRpos = 10*np.log10(potSinal(s_pb)/potSinal(ruido_pb))
print('SNRpos = %.2f dB'%SNRpos)
# -

x_demod = x_demod/abs(x_demod).max(0)
wavfile.write('voz_demodAMSSB.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz_demodAMSSB.wav')

t = np.arange(0, len(x))*1/Fs
plt.figure(figsize =(12,4))
plt.plot(t, x, linewidth = 0.5, label='sinal de voz')
plt.plot(t, x_demod, linewidth = 0.5, label='sinal demodulado')
plt.xlabel('tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(min(t), max(t));
plt.legend();
