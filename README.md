# :page_facing_up: MedShift-SFDA: A Difficulty-graded Benchmark for Source-Free Domain Adaptation in Medical Image Segmentation

### The image below shows a comparison of the segmentation before and after SFDA approval.

<p align="center"><img src="1.png" width="90%"></p>

### The following images show examples and ground truth masks for 16 datasets in MedShift-SFDA.

<p align="center"><img src="2.png" width="90%"></p>

### The image below shows a visualization of inter-dataset domain shift.  

<p align="center"><img src="3.png" width="90%"></p>

### The figure below illustrates the domain shift classification between different modal datasets.

<p align="center"><img src="4.png" width="90%"></p>

### The figure below visualizes the segmentation performance of six methods at different domain shift levels, highlighting the challenges encountered as the difficulty increases.

<p align="center"><img src="5.png" width="90%"></p>

### The table illustrates the key technical aspects and adaptations of different methods.

<p align="center"><img src="6.png" width="90%"></p>

### Clinical significance of this benchmark
```shell
In clinical practice, understanding domain shift severity is essential for deploying trustworthy AI systems across heterogeneous environments. For instance, transferring a segmentation model trained on high-resolution hospital-grade ultrasound to portable devices used in resource-limited clinics constitutes a high-shift scenario, where image quality, anatomical visibility, and patient demographics may all differ. The difficulty-graded benchmark provided by MedShift-SFDA offers valuable guidance for:

1. Model selection based on clinical deployment context

For settings with low domain shift (e.g., same modality, equipment, and anatomy), lightweight methods like ADAMI or DPL are sufficient. In medium shift settings (e.g., cross-center, moderate anatomical variation), robust contrastive and correction-based strategies like PCPDL or CBMT are preferable. For high shift domains (e.g., complex pathology), more advanced models with semantic consistency mechanisms are required.

2. Risk estimation and adaptation planning

Clinicians and engineers can use domain shift metrics (e.g., MMD + t-SNE visualizations) to anticipate adaptation challenges, proactively select suitable methods, or even trigger model revalidation before clinical deployment.

In summary, our benchmark not only evaluates SFDA algorithmic performance but also bridges the gap between method development and real-world healthcare deployment, offering practical insights for building safe, reliable, and generalizable medical AI systems.
```
### Dependency Preparation
```shell
cd MedShift-SFDA
# Python Preparation
conda create -n MedShift-SFDA python=3.8.5
activate MedShift-SFDA
# (torch 1.7.1+cu110) It is recommended to use the conda installation on the Pytorch website https://pytorch.org/
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
### Six Methods for Model Training and Inference
- 1. Download the dataset in the paper and modify the relevant path in the configuration file.
- 2. Source Model Train
   -- We use the code provided by [ProSFDA](https://github.com/ShishuaiHu/ProSFDA) to train the source model. If you want to use our trained source model, please contact me.
- 3. Steps to debug six methods: 
```shell
1. DPL Method:
(1) Source code link: [DPL](https://github.com/cchen-cc/SFDA-DPL)
(2) Please click DPL file
(3) Generation phase: generate target domain pseudo-labels
python generate_pseudo.py
(4) Adaptation stage: source model adapts to the target domain
python train_target.py
2. CBMT Method:
(1) Source code link: [CBMT](https://github.com/lloongx/SFDA-CBMT)
(2) Please click CBMT file
(3) Adaptation stage: source model adapts to the target domain
python train_target.py
3. CPR Method:
(1) Source code link: [CPR](https://github.com/xmed-lab/CPR)
(2) Please click CPR file
(3) Generation phase: generate target domain pseudo-labels
python generate_pseudo.py
(4) Adaptation stage: source model adapts to the target domain
Please run them in order:
python sim_learn.py
python pl_refine.py
python train_target.py
4. PCPDL Method:
(1) Source code link: [PCPDL](https://github.com/M4cheal/PCDCL-SFDA)
(2) Please click PCPDL file
(3) Generation phase: generate target domain pseudo-labels
python generate_pseudo.py
(4) Adaptation stage: source model adapts to the target domain
python train_target.py
5. FSM Method:
(1) Source code link: [FSM](https://github.com/CityU-AIM-Group/SFDA-FSM)
(2) Please click on the MedShift-SFDA file and enter the FSM
(3) Generate source-like images
python domain_inversion.py
(4) Adaptation stage: source model adapts to the target domain
python train_adapt.py 
6. ADAMI Method:
(1) Source code link: [ADAMI](https://github.com/mathilde-b/SFDA)
(2) Please click ADAMI file
(3) Adaptation stage: source model adapts to the target domain
python train_target.py
