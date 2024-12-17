Custom implementation of CycleGan-VC2

Based on:
- PARALLEL-DATA-FREE VOICECONVERSION
 USINGCYCLE-CONSISTENT ADVERSARIALNETWORKS
 Takuhiro Kaneko, Hirokazu Kameoka
 NTTCommunication Science Laboratories, NTT Corporation, Japan

- CYCLEGAN-VC2:
 IMPROVEDCYCLEGAN-BASEDNON-PARALLELVOICECONVERSION
 Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Nobukatsu Hojo
 NTTCommunication Science Laboratories, NTT Corporation, Japan

- CycleGAN-VC3:
 Examining and Improving CycleGAN-VCs for Mel-spectrogram Conversion
 Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Nobukatsu Hojo
 NTTCommunication Science Laboratories, NTT Corporation, Japan

- STARGAN-VC:NON-PARALLELMANY-TO-MANYVOICECONVERSION
 WITHSTARGENERATIVEADVERSARIALNETWORKS
 Hirokazu Kameoka, Takuhiro Kaneko, Kou Tanaka, Nobukatsu Hojo
 NTTCommunication Science Laboratories, NTT Corporation, Japan

- StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for
 Natural-Sounding Voice Conversion
 Yinghao Aaron Li, Ali Zare, Nima Mesgarani
 Department of Electrical Engineering, Columbia University, USA

Training data: https://datashare.ed.ac.uk/handle/10283/3061

Folder results/samples contains examples of conversion during different stages of training between VCC2SF1 and VCC2SM2.
The final result is after 200,000 iterations.


Results:

Converted voice contains reproduces characteristic speech style of the target speaker. MCD and MSD statistics show that conversion works pretty well, however the pitch conversion using logarithm Gaussian normalized transformation is not enough to have a decent sounding sample. Since this is not enough i will attempt to make a CycleGan-VC3, because its pitch conversion is with use of GAN, also there is still a room for imporvement of produced MCEP(in VC3 mel spectrograms).
