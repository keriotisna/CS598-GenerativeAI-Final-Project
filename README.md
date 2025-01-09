# Project Overview
---

An implementation of a U-Net diffusion architecture for stylized handwriting generation. All models are trained on subsets of the EMNIST dataset by using unsupervised learning methods to extract styled samples, on which models are trained to generate stylized samples. 

![ep-149_w-2_nc-16_ts-500](https://github.com/user-attachments/assets/24b19db3-0c09-4572-b41c-a081244877ec)
- Stylized samples of the letter 'a' being generated from a trained model.


![EAugmented500](https://github.com/user-attachments/assets/d767ad7d-72fd-445b-8741-034876131c56)
- An example of extracted styles obtained by using UMAP as a non-linear dimensionality reduction technique.

<img width="604" alt="Pasted image 20250108181353" src="https://github.com/user-attachments/assets/e02c3675-2705-4fce-97c2-a0578e94fa91" />

- A zoomed in picture of the above image focusing on 4 distinct 'styles'.

<img width="871" alt="Pasted image 20250108175918" src="https://github.com/user-attachments/assets/188c48c7-5591-4309-9448-344aa8d4796e" />

- Overview of the U-Net architecture. Stylized samples are first encoded to a latent space with fewer dimensions, then new samples are generated using a diffusion process from this latent space.
