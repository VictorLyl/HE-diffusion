# HE-Diffusion

[HE-Diffusion]is a privacy-preserving latent text-to-image diffusion model based on [Stable diffusion](https://huggingface.co/CompVis/stable-diffusion). It works for protecting users' input prompts and output images during inference.

  
## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created similar with stable diffusion
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
``` 

Besides requirements by stable diffusion, [TenSeal](https://github.com/OpenMined/TenSEAL/tree/main) is also required in He-Diffusion for homomorphic encryption. It can be simply installed by:
```
pip install tenseal
```

#### Reference Sampling Script

User can simply sample with
```
python scripts/enc_txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
```

By default, this uses a guidance scale of `--scale 7.5`, [Katherine Crowson's implementation](https://github.com/CompVis/latent-diffusion/pull/51) of the [PLMS](https://arxiv.org/abs/2202.09778) sampler, 
and renders images of size 512x512 (which it was trained on) in 50 steps. All supported arguments are listed below (type `python scripts/enc_txt2img.py --help`).


```commandline
usage: enc_txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--skip_grid] [--skip_save] [--ddim_steps DDIM_STEPS] [--plms] [--laion400m] [--fixed_code] [--ddim_eta DDIM_ETA]
                  [--n_iter N_ITER] [--H H] [--W W] [--C C] [--f F] [--n_samples N_SAMPLES] [--n_rows N_ROWS] [--scale SCALE] [--from-file FROM_FILE] [--config CONFIG] [--ckpt CKPT]
                  [--seed SEED] [--precision {full,autocast}]

optional arguments:
  -h, --help            show this help message and exit
  --prompt [PROMPT]     the prompt to render
  --outdir [OUTDIR]     dir to write results to
  --skip_grid           do not save a grid, only individual samples. Helpful when evaluating lots of samples
  --skip_save           do not save individual samples. For speed measurements.
  --ddim_steps DDIM_STEPS
                        number of ddim sampling steps
  --plms                use plms sampling
  --laion400m           uses the LAION400M model
  --fixed_code          if enabled, uses the same starting code across samples
  --ddim_eta DDIM_ETA   ddim eta (eta=0.0 corresponds to deterministic sampling
  --n_iter N_ITER       sample this often
  --H H                 image height, in pixel space
  --W W                 image width, in pixel space
  --C C                 latent channels
  --f F                 downsampling factor
  --n_samples N_SAMPLES
                        how many samples to produce for each given prompt. A.k.a. batch size
  --n_rows N_ROWS       rows in the grid (default: n_samples)
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --from-file FROM_FILE
                        if specified, load prompts from this file
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of model
  --seed SEED           the seed (for reproducible sampling)
  --precision {full,autocast}
                        evaluate at this precision
```
Note: The inference config for all v1 versions is designed to be used with EMA-only checkpoints. 
For this reason `use_ema=False` is set in the configuration, otherwise the code will try to switch from
non-EMA to EMA weights. If you want to examine the effect of EMA vs no EMA, we provide "full" checkpoints
which contain both types of weights. For these, `use_ema=False` will load and use the non-EMA weights.

## Content under development and future work
Here we only support text to image task, which is most relevantly used. In the future we will try to propose more complete tasks.


## 问题
### 部署问题1
创建conda环境报错
> ERROR: Command errored out with exit status 128: git clone -q https://github.com/CompVis/taming-transformers.git /mnt/HE-diffusion/src/taming-transformers Check the logs for full command output.

解决：手动下载两个包，并进入仓库目录执行安装

> 修改[environment.yaml](environment.yaml)以下代码
> - -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
> - -e git+https://github.com/openai/CLIP.git@main#egg=clip
> 
> 改为（path改为实际路径）
> - -e /path/HE-diffusion/src/taming-transformers
> - -e /path/HE-diffusion/src/CLIP
>
> pip install -e .

### 部署问题2
Hugging Face无法连接
> OSError: We couldn't connect to 'https://huggingface.co' to load this model, couldn't find it in the cached files and it looks like CompVis/stable-diffusion-safety-checker is not the path to a directory containing a preprocessor_config.json file.

> 解决：
> 
> 临时设置：
> - export HF_ENDPOINT=https://hf-mirror.com
> 
> 永久设置：
> - vim ~/.bashrc
> - 最后一行添加：export HF_ENDPOINT=https://hf-mirror.com
> - source ~/.bashrc

### 部署问题3

> FileNotFoundError: [Errno 2] No such file or directory: 'models/ldm/stable-diffusion-v1/model.ckpt'

> 解决：
> 
> 本地下载至[/HE-Diffusion/models/ldm/stable-diffusion-v1/]目录下：https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/blob/main/sd-v1-4-full-ema.ckpt
>
> 并修改文件名：mv sd-v1-4-full-ema.ckpt model.ckpt

### 执行问题1

> RuntimeError: expected scalar type BFloat16 but found Float

> 解决（使用GPU）：
> 修改[scripts/enc_txt2img.py]的250-251、301-302行代码
> - device = torch.device("cpu") # 250
> - #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # 251
> - #with precision_scope("cuda"): # 301
> - with precision_scope("cpu"): # 302
> 
> 改为
> - #device = torch.device("cpu") # 250
> - device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # 251
> - #with precision_scope("cuda"): # 301
> - with precision_scope("cpu"): # 302
