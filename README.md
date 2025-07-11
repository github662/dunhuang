# MP-WSA
Code for the paper titled "Mural Image Inpainting via Mamba Prior and Wavelet-Sparse Transformer".


<br>
This is the code for amba Prior and Wavelet-Sparse Transformer (MP-WSA) to reconstruct corrupted image. Given one image and mask, the proposed **MP-WSA** model is able to reconstruct masked regions.

## Illustration of MP-WSA
![](https://github.com/huangwenwenlili/wsa-former/blob/main/images/wsa-former-architecture.png)

(1) In the field of mural heritage conservation and presentation, the inpainting of damaged mural images has gradually emerged as a prominent research focus. To address the challenges present in existing mural image inpainting algorithms, particularly the lack of global coherence and insufficient detail restoration in complex regions of the reconstructed images, this paper proposes a Mamba Prior and Wavelet-Sparse Transformer-based approach to enhance the quality of inpainting.

(2) This paper presents a dual-branch mural image inpainting framework that integrates Mamba priors and wavelet sparse transformations. The prior-guided branch leverages the Mamba Prior module to model long-range dependencies and global semantic contexts, extracting coherent global prior features to guide the inpainting process and overcoming the limitations of static prior information. The backbone inpainting branch incorporates a Wavelet Sparse Attention module, employing multi-scale sparse modeling to effectively capture essential texture details while balancing detail extraction and computational efficiency. Additionally, a Gated Dynamic Tanh Feedforward module enhances nonlinear attention representation, thereby alleviating the suppression of fine details. During the upsampling process, prior-guided features are fused to improve structural inpainting and detail coherence, resulting in realistic and natural inpainting outcomes. Extensive experiments conducted on our custom-built dataset demonstrate that the proposed method achieves superior performance across multiple evaluation metrics, including PSNR, SSIM, LPIPS, and FID, while maintaining an efficient computational profile. This approach holds significant potential for the digital inpainting of murals, making a meaningful contribution to cultural heritage conservation and dissemination.   


# Getting started  
## Installation
This code was tested with Pytoch 1.8.1 CUDA 11.4, Python 3.8 and Ubuntu 18.04
   
- Create conda environment:

```
conda create -n inpainting-dunhuang python=3.8
conda deactivate
conda activate inpainting-dunhuang 
pip install visdom dominate
```
- Clone this repo:1

```
git clone https://github.com/github662/dunhuang
cd dunhuang
```

- Pip install libs:

```
pip install -r requirements.txt
```

## Datasets
- ```Dunhuang Art Digital Restoration Dataset```: It contains a dataset of Dunhuang murals from natural digital images.
[Paris](https://github.com/github662/-)

## Train
- Train the model using input images and masks with a resolution of 256×256. During the training phase, randomly generated irregular masks are applied to artificially corrupt the images, simulating missing regions for the inpainting task.
```
python train.py --name dunhuang  --img_file /home/hwl/hwl/datasets/paris/paris_train_original/ --niter 261000 --batchSize 4 --lr 1e-4 --gpu_ids 1 --no_augment --no_flip --no_rotation 
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **2 and 4 . random irregular mask**.
- ```--lr``` is learn rate, train scratch is 1e-4, finetune is 1e-5.

## Testing

- Test the model. Input images and masks resolution are 256*256. In the testing, we use [irregular mask dataset](https://github.com/NVIDIA/partialconv) to evaluate different ratios of corrupted region images.

```
python test.py  --name dunhuang --checkpoints_dir ./checkpoints/checkpoint_dunhuang --gpu_ids 0 --img_file your_image_path --mask_file your_mask_path --batchSize 1 --results_dir your_image_result_path
```
- Set ```--mask_type``` in options/base_options.py to test various masks. ```--mask_file``` path is needed for **3. external irregular mask**,
- The default results will be saved under the *results* folder. Set ```--results_dir``` for a new path to save the result.


## Example Results
- **Comparison results of softmax-based attention and our proposed Wsa-attention**
![](https://github.com/huangwenwenlili/wsa-former/blob/main/images/wsa-intr.png)

(a) Attention values were computed for one feature channel using the softmax and ReLU functions. We found that the ReLU function generated attention values that were more focused on essential contexts, compared to the dense attention values obtained from the softmax function. 

(b) We compared the inpainting results obtained using these two attention mechanisms. Our Wsa-attention approach yielded superior results, as indicated by the improved completion of the building window in the Wsa-attention completion image, and the smaller FID value obtained by this method. Lower values of FID are indicative of better performance

## Reference Codes
- https://github.com/lyndonzheng/Pluralistic-Inpainting
- https://github.com/NVIDIA/partialconv

## Citation

If you use this code for your research, please cite our paper.

```
@article{huang2023wsaformer,
  title={Mural Image Inpainting via Mamba Prior and Wavelet-Sparse Transformer},
  author={Xu,Zhigang and Li，Jie},
  journal={},
  volume={},
  pages={},
  year={}
}
```
