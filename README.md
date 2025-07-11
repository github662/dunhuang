# MP-WSA
Code for the paper titled "Mural Image Inpainting via Mamba Prior and Wavelet-Sparse Transformer".


<br>
"This algorithm implements a mural image inpainting system based on the Mamba Prior and Wavelet-Sparse Transformer (MP-WSA). Given an input mural image and its corresponding damage region mask, the proposed MP-WSA model performs heritage-grade digital restoration of the occluded areas in cultural relic murals."


## Illustration of MP-WSA
![](https://github.com/github662/dunhuang/blob/main/images/Overall_Network_Architecture.svg)

(1) In the field of mural heritage conservation and presentation, the inpainting of damaged mural images has gradually emerged as a prominent research focus. To address the challenges present in existing mural image inpainting algorithms, particularly the lack of global coherence and insufficient detail restoration in complex regions of the reconstructed images, this paper proposes a Mamba Prior and Wavelet-Sparse Transformer-based approach to enhance the quality of inpainting.

(2) This paper presents a dual-branch mural image inpainting framework that integrates Mamba priors and wavelet sparse transformations. The prior-guided branch leverages the Mamba Prior module to model long-range dependencies and global semantic contexts, extracting coherent global prior features to guide the inpainting process and overcoming the limitations of static prior information. The backbone inpainting branch incorporates a Wavelet Sparse Attention module, employing multi-scale sparse modeling to effectively capture essential texture details while balancing detail extraction and computational efficiency. Additionally, a Gated Dynamic Tanh Feedforward module enhances nonlinear attention representation, thereby alleviating the suppression of fine details. During the upsampling process, prior-guided features are fused to improve structural inpainting and detail coherence, resulting in realistic and natural inpainting outcomes. Extensive experiments conducted on our custom-built dataset demonstrate that the proposed method achieves superior performance across multiple evaluation metrics, including PSNR, SSIM, LPIPS, and FID, while maintaining an efficient computational profile. This approach holds significant potential for the digital inpainting of murals, making a meaningful contribution to cultural heritage conservation and dissemination.   


# Getting started  
## Installation
This code was tested with Pytoch 1.8.1 CUDA 11.4, Python 3.8 and Ubuntu 22.04
   
- Create conda environment: 

```
conda create -n inpainting-dunhuang python=3.8
conda deactivate
conda activate inpainting-dunhuang 
pip install visdom dominate
```
- Clone this repo:

```
git clone https://github.com/github662/dunhuang
cd dunhuang
```

- Pip install libs:

```
pip install -r requirements.txt
```

## Datasets
- ```Dunhuang mural color painting replication dataset```: It contains a dataset of Dunhuang murals from natural digital images.
- https://github.com/github662/-

## Train
- Train the model using input images and masks with a resolution of 256×256. During the training phase, randomly generated irregular masks are applied to artificially corrupt the images, simulating missing regions for the inpainting task.
```
python train.py --name dunhuang  --img_file your_image_path --mask_file your_mask_path --niter 300000 --batchSize 8 --lr 1e-4 --gpu_ids 1 --no_augment --no_flip --no_rotation 
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **2 and 4 . random irregular mask**.
- ```--lr``` is learn rate, train scratch is 1e-4, finetune is 1e-5.

## Testing

- Test the model. Input images and masks resolution are 256*256. In the testing, we use [irregular mask dataset](https://github.com/NVIDIA/partialconv) to evaluate different ratios of corrupted region images.

```
python test.py  --name dunhuang --checkpoints_dir ./checkpoints/checkpoint_dunhuang --gpu_ids 1 --img_file your_image_path --mask_file your_mask_path --batchSize 2 --results_dir your_image_result_path
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **3. external irregular mask**,
- The default results will be saved under the *results* folder. Set ```--results_dir``` for a new path to save the result.


## Example Results
- ** Mural_Inpainting_under_Simulated_Damage **
![](https://github.com/github662/dunhuang/blob/main/images/Mural_Inpainting_under_Simulated_Damage.svg)

- Figs.(c)-(h) display inpainting results from different methods: RFR exhibits edge distortion, MISF shows insufficient structure-texture integration, CSWT produces partial blurring, T-former retains localized artifacts, U2FAN introduces visible artifacts, and HINT demonstrates minor inconsistencies. Our method (Fig. 6(i)) outperforms all comparisons in structural continuity, semantic coherence, and natural texture synthesis, with the wavelet sparse attention module notably enhancing cross-scale perception. However, room for improvement remains in handling large missing regions.
## Reference Codes
- https://github.com/huangwenwenlili/spa-former
- https://github.com/NVIDIA/partialconv

## Citation

If you use this code for your research, please cite our paper.

```
@article{xu2025mp-wsa,
  title={Mural Image Inpainting via Mamba Prior and Wavelet-Sparse Transformer},
  author={Xu,Zhigang and Li，Jie},
  journal={},
  volume={},
  pages={},
  year={}
}
```
