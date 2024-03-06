# MAP2V

Validating Privacy-Preserving Face Recognition under a Minimum Assumption
To be presented in the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024

![Overview of the proposed Map2V. ](/imgs/pipeline.png)

## Abstract

The widespread use of cloud-based face recognition technology raises privacy concerns, as unauthorized access to face images can expose personal information or be exploited for fraudulent purposes.  In response, privacy-preserving face recognition (PPFR) schemes  have emerged to hide visual information and impede recovery. However, their privacy validation methods, built on unrealistic assumptions, raise doubts about proclaimed face privacy preservation capabilities. In this paper, we introduce a general privacy-preserving validation version under the novel concept of minimum assumption coined Map$^2$V. This is the first exploration of formulating a privacy validation method utilizing deep image priors and zeroth-order gradient estimation, with the potential to serve as a generalizable tool for PPFR evaluation. Building upon Map$^2$V, we comprehensively validate the anti-privacy inference capability of PPFRs through a combination of human and machine vision. The results of experiments and analysis demonstrate the effectiveness and generalizability of the proposed Map$^2$V, showcasing its superiority over native privacy validation methods from PPFR works of literature. Additionally, this work exposes privacy vulnerabilities in evaluated state-of-the-art PPFR schemes, laying the foundation for the subsequent effective proposal of countermeasures.

## Requirements
-   PyTorch 1.9.1
-   Torchvision 0.10.1
-   CUDA 10.1/10.2

## Setup

Download pretrained privacy-preserving face recognition model and weights:
[DuetFace](https://github.com/Tencent/TFace/tree/master/recognition/tasks/duetface)
[DCTDP](https://github.com/Tencent/TFace/tree/master/recognition/tasks/dctdp)
[PartialFace](https://github.com/Tencent/TFace)

## Download LFW and CelebA datasets:

[LFW](https://drive.google.com/file/d/1lckCEDPjOFAyJRjpdWnfseqI50_yEXAW/view)

CelebA

After downloading, change the paths in `dataset/dataset_conf.yaml` and `weight` in `encoder/encoder_conf.yaml` accordingly.

## Download the latents from the constructed Prior Space:

latents

## Usage

After the setup is done, simply run reconstruct.py

## Experimental Results
Privacy scores (%) against different validation systems on  LFW and CelebA dataset under 1k1c settings.
![输入图片说明](/imgs/results1.png)
Examples of reconstructed faces for three SOTAs under 1k1c settings.
![输入图片说明](/imgs/results.png)

#  Citation
If you find this code useful in your research, please consider citing us.

