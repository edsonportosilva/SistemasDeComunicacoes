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
# <h1>Sumário<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Carrega-sinais-de-áudio" data-toc-modified-id="Carrega-sinais-de-áudio-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Carrega sinais de áudio</a></span><ul class="toc-item"><li><span><a href="#Plota-sinais-no-domínio-do-tempo" data-toc-modified-id="Plota-sinais-no-domínio-do-tempo-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Plota sinais no domínio do tempo</a></span></li><li><span><a href="#Plota-densidades-espectrais-de-potência" data-toc-modified-id="Plota-densidades-espectrais-de-potência-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Plota densidades espectrais de potência</a></span></li></ul></li><li><span><a href="#Modulação-AM-DSB-SC" data-toc-modified-id="Modulação-AM-DSB-SC-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Modulação AM-DSB-SC</a></span><ul class="toc-item"><li><span><a href="#Demodulação-síncrona-AM-DSB-SC" data-toc-modified-id="Demodulação-síncrona-AM-DSB-SC-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Demodulação síncrona AM-DSB-SC</a></span></li><li><span><a href="#Áudio-demodulado-AM-DSB-SC" data-toc-modified-id="Áudio-demodulado-AM-DSB-SC-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Áudio demodulado AM-DSB-SC</a></span></li></ul></li><li><span><a href="#Modulação-AM-DSB" data-toc-modified-id="Modulação-AM-DSB-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Modulação AM-DSB</a></span><ul class="toc-item"><li><span><a href="#Demodulação-por-envoltória-AM-DSB" data-toc-modified-id="Demodulação-por-envoltória-AM-DSB-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Demodulação por envoltória AM-DSB</a></span></li><li><span><a href="#Áudio-demodulado-AM-DSB" data-toc-modified-id="Áudio-demodulado-AM-DSB-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Áudio demodulado AM-DSB</a></span></li></ul></li><li><span><a href="#Modulação-AM-SSB" data-toc-modified-id="Modulação-AM-SSB-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Modulação AM-SSB</a></span><ul class="toc-item"><li><span><a href="#USB-e-LSB" data-toc-modified-id="USB-e-LSB-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>USB e LSB</a></span></li><li><span><a href="#Demodulação-síncrona-AM-SSB" data-toc-modified-id="Demodulação-síncrona-AM-SSB-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Demodulação síncrona AM-SSB</a></span></li><li><span><a href="#Áudio-demodulado-AM-SSB" data-toc-modified-id="Áudio-demodulado-AM-SSB-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Áudio demodulado AM-SSB</a></span></li></ul></li><li><span><a href="#Modulação-AM-ISB" data-toc-modified-id="Modulação-AM-ISB-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Modulação AM-ISB</a></span><ul class="toc-item"><li><span><a href="#Demodulação-síncrona-AM-ISB" data-toc-modified-id="Demodulação-síncrona-AM-ISB-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Demodulação síncrona AM-ISB</a></span></li><li><span><a href="#Áudio-demodulado-AM-ISB-(sem-filtragem-passa-faixa)" data-toc-modified-id="Áudio-demodulado-AM-ISB-(sem-filtragem-passa-faixa)-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Áudio demodulado AM-ISB (sem filtragem passa-faixa)</a></span></li><li><span><a href="#Filtragem-passa-faixa" data-toc-modified-id="Filtragem-passa-faixa-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Filtragem passa-faixa</a></span></li><li><span><a href="#Áudio-demodulado-AM-ISB-(com-filtragem-passa-faixa)" data-toc-modified-id="Áudio-demodulado-AM-ISB-(com-filtragem-passa-faixa)-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Áudio demodulado AM-ISB (com filtragem passa-faixa)</a></span></li></ul></li><li><span><a href="#Modulação-QAM" data-toc-modified-id="Modulação-QAM-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modulação QAM</a></span><ul class="toc-item"><li><span><a href="#Demodulação-síncrona-QAM" data-toc-modified-id="Demodulação-síncrona-QAM-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Demodulação síncrona QAM</a></span></li><li><span><a href="#Áudio-demodulado-QAM" data-toc-modified-id="Áudio-demodulado-QAM-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Áudio demodulado QAM</a></span></li></ul></li></ul></div>

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

m1 = m1-np.mean(m1)
m2 = m2-np.mean(m2)
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

# ## Modulação AM-DSB-SC

# +
# Exemplo 1 (AM-DSB-SC):

Fs     = 48e3     # frequência de amostragem do sinal de áudio
fc_tx  = 10e3     # frequência da portadora
B_sig  = 4e3      # largura de banda do sinal de áudio
ntaps  = 4096+1   # número de coeficientes dos filtros
SNR    = 50       # SNR desejada em dB
π      = np.pi
θ      = 0

g = lowpass_firwin(ntaps, B_sig, Fs)
w, G = freqz(g, fs=Fs, worN=4096)

# plota o valor absoluto das resposta em frequência do filtro
plt.plot(w, 10*np.log10(np.abs(G)), linewidth=2, label = 'G(f)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Frequency response of the filter')
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

# +
wavfile.write('voz1_filtrada.wav', int(Fs), x.astype(np.float32))

IPython.display.Audio('voz1_filtrada.wav')

# +
# Modula sinal de voz DSB-SC e plota espectro do sinal modulado:
s_dsb = x*np.cos(2*π*fc_tx*t + θ)

Ps = potSinal(s_dsb) # calcula potência do sinal modulado

t_plot = np.arange(1500)*1/Fs

plt.figure(figsize =(12,4))
plt.plot(t_plot, s_dsb[0:t_plot.size], linewidth=0.8, label='sinal modulado AM-DSB-SC')
plt.xlabel('tempo(s)')
plt.ylabel('$s(t)$')
plt.xlim(np.min(t_plot), np.max(t_plot))
plt.legend()
plt.grid()

plt.figure(figsize =(12,4))
plt.psd(s_dsb, Fs=Fs, label='DEP do sinal AM-DSB-SC',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Demodulação síncrona AM-DSB-SC

# +
s_demod  = s_dsb*np.cos(2*π*fc_tx*t) # demodulação síncrona
x_demod  = filterNoDelay(g, s_demod)    # filtragem passa-baixa

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Áudio demodulado AM-DSB-SC

# +
x_demod = x_demod/abs(x_demod).max(0)

wavfile.write('voz1_demodAMDSBSC.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz1_demodAMDSBSC.wav')
# -

# ## Modulação AM-DSB

# +
ka = 1 # índice de modulação AM-DSB

x = x/np.max(np.abs(x))

# Modula sinal de voz DSB e plota espectro do sinal modulado:
s_dsb = (1 + ka*x)*np.cos(2*π*fc_tx*t + θ)

Ps = potSinal(s_dsb) # calcula potência do sinal modulado

t_plot = np.arange(1500)*1/Fs

plt.figure(figsize =(12,4))
plt.plot(t_plot, s_dsb[0:t_plot.size], linewidth=0.8, label='sinal modulado AM-DSB')
plt.xlabel('tempo(s)')
plt.ylabel('$s(t)$')
plt.xlim(np.min(t_plot), np.max(t_plot))
plt.legend()
plt.grid()

plt.figure(figsize =(12,4))
plt.psd(s_dsb, Fs=Fs, label='DEP do sinal AM-DSB-SC',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Demodulação por envoltória AM-DSB

# +
s_demod  = np.abs(s_dsb)  # demodulação por envoltória
s_demod  = s_demod - np.mean(s_demod)
x_demod  = filterNoDelay(g, s_demod)  # filtragem passa-baixa

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Áudio demodulado AM-DSB

# +
x_demod = x_demod/abs(x_demod).max(0)

wavfile.write('voz_demodAMDSB.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz_demodAMDSB.wav')
# -

# ## Modulação AM-SSB

# ### USB e LSB

# +
x1 = filterNoDelay(g, m1)
x2 = filterNoDelay(g, m2)

# Modula sinal de voz SSB e plota epectro do sinal modulado:
s_ssb = 1/np.sqrt(2)*( x1*np.cos(2*π*fc_tx*t) - hilbert(x1).imag*np.sin(2*π*fc_tx*t) )

Ps = potSinal(s_ssb)

t_plot = np.arange(1500)*1/Fs

plt.figure(figsize =(12,4))
plt.plot(t_plot, s_ssb[0:t_plot.size], linewidth=0.8, label='sinal modulado AM-SSB')
plt.xlabel('tempo(s)')
plt.ylabel('$s(t)$')
plt.xlim(np.min(t_plot), np.max(t_plot))
plt.legend()
plt.grid()

plt.figure(figsize =(12,4))
plt.psd(s_ssb, Fs=Fs, label='DEP do sinal AM-SSB',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Demodulação síncrona AM-SSB

# +
s_demod  = s_ssb*np.cos(2*π*fc_tx*t)
x_demod  = filterNoDelay(g, s_demod)

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Áudio demodulado AM-SSB

# +
x_demod = x_demod/abs(x_demod).max(0)

wavfile.write('voz_demodAMSSB.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz_demodAMSSB.wav')
# -

t = np.arange(0, len(x))*1/Fs
plt.figure(figsize =(12,4))
plt.plot(t, x, linewidth = 0.5, label='sinal de voz')
plt.plot(t, x_demod, linewidth = 0.5, label='sinal demodulado')
plt.xlabel('tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.xlim(min(t), max(t));
plt.legend();

# ## Modulação AM-ISB

# +
# Modula sinais de voz e plota epectro do sinal modulado:
s_ssb_1 = 1/np.sqrt(2)*( x1*np.cos(2*π*fc_tx*t) + hilbert(x1).imag*np.sin(2*π*fc_tx*t) )

s_ssb_2 = 1/np.sqrt(2)*( x2*np.cos(2*π*fc_tx*t) - hilbert(x2).imag*np.sin(2*π*fc_tx*t) )

s_isb = s_ssb_1 + s_ssb_2

Ps = potSinal(s_isb)

t_plot = np.arange(1500)*1/Fs

plt.figure(figsize =(12,4))
plt.plot(t_plot, s_isb[0:t_plot.size], linewidth=0.8, label='sinal modulado AM-ISB')
plt.xlabel('tempo(s)')
plt.ylabel('$s(t)$')
plt.xlim(np.min(t_plot), np.max(t_plot))
plt.legend()
plt.grid()

plt.figure(figsize =(12,4))

plt.psd(s_ssb_1, Fs=Fs,label='DEP do sinal AM-SSB-LSB',\
        sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(s_ssb_2, Fs=Fs,label='DEP do sinal AM-SSB-USB',\
        sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal

plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)

plt.figure(figsize =(12,4))
plt.psd(s_isb, Fs=Fs, color='black', label='DEP do sinal AM-ISB',\
        sides='twosided', linewidth=0.5, NFFT=4096); # estima DEP do sinal

plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Demodulação síncrona AM-ISB

# +
s_demod  = s_isb*np.cos(2*π*fc_tx*t)
x_demod  = filterNoDelay(g, s_demod)

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Áudio demodulado AM-ISB (sem filtragem passa-faixa)

# +
x_demod = x_demod/abs(x_demod).max(0)

wavfile.write('voz_demodAMISB.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz_demodAMISB.wav')
# -

# ### Filtragem passa-faixa

# +
# frequências de corte do filtro passa-faixa:
lowcut  = fc_tx 
highcut = fc_tx + B_sig
ntaps  = 4096+1   # número de coeficientes dos filtros

h = bandpass_firwin(ntaps, lowcut, highcut, Fs)
w, H = freqz(h, fs=Fs, worN=4096)

# plota o valor absoluto das resposta em frequência do filtro
plt.plot(w, 10*np.log10(np.abs(H)), linewidth=1, label = 'H(f)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Frequency response of the filter')
plt.legend()
plt.grid(True)
plt.xlim(min(w),max(w));

# +
s_isb_filt  = filterNoDelay(h, s_isb)

s_demod  = s_isb_filt*np.cos(2*π*fc_tx*t)

x_demod  = filterNoDelay(g, s_demod)

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Áudio demodulado AM-ISB (com filtragem passa-faixa)

# +
x_demod = x_demod/abs(x_demod).max(0)

wavfile.write('voz_demodAMISBfilt.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz_demodAMISBfilt.wav')
# -

# ## Modulação QAM

# +
x1 = filterNoDelay(g, m1)
x2 = filterNoDelay(g, m2)

# Modula sinal de voz SSB e plota epectro do sinal modulado:
s_qam = x1*np.cos(2*π*fc_tx*t) + x2*np.sin(2*π*fc_tx*t)

Ps = potSinal(s_ssb)

t_plot = np.arange(1500)*1/Fs

plt.figure(figsize =(12,4))
plt.plot(t_plot, s_qam[0:t_plot.size], linewidth=0.8, label='sinal modulado QAM')
plt.xlabel('tempo(s)')
plt.ylabel('$s(t)$')
plt.xlim(np.min(t_plot), np.max(t_plot))
plt.legend()
plt.grid()

plt.figure(figsize =(12,4))
plt.psd(s_qam, Fs=Fs, label='DEP do sinal QAM',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)

# -

# ### Demodulação síncrona QAM

# +
s_demod  = s_qam*np.sin(2*π*fc_tx*t)

x_demod  = filterNoDelay(g, s_demod)

plt.figure(figsize =(12,4))
plt.psd(s_demod, Fs=Fs, label='DEP do sinal após o mixer',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.psd(x_demod, Fs=Fs, label='DEP do sinal após o fpb',sides='twosided', linewidth=0.8, NFFT=4096); # estima DEP do sinal
plt.legend();
plt.xlim(-Fs/2,Fs/2);
plt.grid(True)
# -

# ### Áudio demodulado QAM

# +
x_demod = x_demod/abs(x_demod).max(0)

wavfile.write('voz_demodQAM.wav', int(Fs), x_demod.astype(np.float32))

IPython.display.Audio('voz_demodQAM.wav')
