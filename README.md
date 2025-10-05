# 1. File Preparation

## Soil types.zip
Contains the images of each soil type.
### Soil Types and Characteristics

| Order | Soil Type | Color / Common Name | Main Characteristics | Fertility / Ideal Use |
| --- | --- | --- | --- | --- |
| 1 | **Black Soil (Solo Preto)** | Black soil / Regossolo / Terra roxa | - Rich in **clay, calcium, iron, and magnesium**.  - High **moisture retention**.  - Rich in **nutrients (K, Ca, Mg)**.  - pH **neutral to slightly alkaline**. | Excellent for **wheat, corn, cotton, and barley**. **Most fertile**. |
| 2 | **Brown Soil (Solo Marrom)** | Brown soil / Margo | - Good **aeration and drainage**.  - Contains **moderate organic matter**.  - Neutral pH. | Good for **legumes, potatoes, corn, and fruits**. |
| 3 | **Red Soil (Solo Vermelho)** | Rich in iron oxides | - Low humus content.  - **Acidic** pH.  - Low water retention capacity. | Used with **fertilization** for **peanut, corn, sorghum, and cotton**. |
| 4 | **Dark Black Soil (Solo Preto Escuro / Basaltic)** | Very dark soil, rich in iron and magnesium | - High **moisture retention**.  - Rich in minerals but **poor in phosphorus**.  - Alkaline pH. | Suitable for **cotton, sugarcane, and sunflower**. |
| 5 | **Yellow Soil (Solo Amarelo)** | Light yellow, found in tropical areas | - Rich in **hydrated iron oxides**.  - **Poor in nutrients and organic matter**.  - Low natural fertility. | Requires **pH correction and fertilization**; used for **cassava and grasses**. |

## dados.csv
CSV file with the following columns:

- **image_path**  
  Path to the soil images in the dataset.

- **text**  
  Contains the soil characteristics and chemical parameters for each sample.

---

# 2. Ockham's Razor Theorem

## Haralick (Pre-processing)

### Introduction

The main **Haralick features** are sets of statistical properties extracted from the **Gray-Level Co-occurrence Matrix (GLCM)**.

These features describe the **texture** of an image — the pattern of gray-level or color variations in a region — and are widely used in **image analysis**, **computer vision**, **pattern classification**, **segmentation**, **medical imaging**, and **remote sensing**.


### How It Works

To implement Haralick features, it is necessary to construct a **GLCM (Gray-Level Co-occurrence Matrix)**, which describes how gray levels occur in pairs within an image at specific **directions** and **distances**.

It counts how often a pixel with a certain intensity value (**i**) occurs adjacent to another pixel with value (**j**) at a given distance and direction.

From this matrix, several **statistical measures** are computed to describe the image texture.

---

### Main Haralick Features (14 Original Texture Properties)

| No. | Property Name | Formula / Description | Interpretation |
| --- | --- | --- | --- |
| 1 | **Angular Second Moment (ASM)** or **Energy** | ∑ᵢⱼ P(i,j)² | Measures **uniformity** of the texture. High values → homogeneous texture. |
| 2 | **Contrast** | ∑ₙ₌₀ⁿ₍g₋₁₎ n² ∑ᵢⱼ P(i,j) | Measures **intensity contrast** between a pixel and its neighbor. |
| 3 | **Correlation** | Measures **linear dependency** of gray levels between neighboring pixels. | High values → strong correlation between pixel intensities. |
| 4 | **Variance** | Variance of gray levels weighted by their co-occurrence. | Measures **gray-level dispersion**. |
| 5 | **Inverse Difference Moment (Homogeneity)** | ∑ᵢⱼ P(i,j) / [1 + (i−j)²] | Measures **local similarity** — high values indicate smooth regions. |
| 6 | **Sum Average** | ∑ₖ k · p₍x+y₎(k) | Average of the sum distribution — describes overall brightness. |
| 7 | **Sum Variance** | Variance of the sum distribution. | Measures **overall variation** in gray levels. |
| 8 | **Sum Entropy** | −∑ₖ p₍x+y₎(k) log p₍x+y₎(k) | Measures **randomness** of gray-level sums. |
| 9 | **Entropy** | −∑ᵢⱼ P(i,j) log P(i,j) | Measures **texture complexity**. High values → disordered texture. |
| 10 | **Difference Variance** | Variance of the difference distribution p₍x−y₎(k). | Measures **variation between neighboring intensities**. |
| 11 | **Difference Entropy** | −∑ₖ p₍x−y₎(k) log p₍x−y₎(k) | Measures **randomness between different gray levels**. |
| 12 | **Information Measure of Correlation 1 (IMC1)** | Based on joint entropy. | Quantifies **statistical dependency** between pixels. |
| 13 | **Information Measure of Correlation 2 (IMC2)** | Variant of IMC1. | Another measure of **pixel interdependence**. |
| 14 | **Maximal Correlation Coefficient** | Computed from the eigenvalues of the GLCM. | Measures **nonlinear correlation** between gray levels. |

## X-means with BIC

The **X-means algorithm** is an extension of **K-means** that solves its main limitation — the need to **manually define the number of clusters** before running the algorithm.

X-means automatically determines the **optimal number of clusters** using a statistical method called the **Bayesian Information Criterion (BIC)**.

---

### Simplified BIC Formula

BIC = ln(L) - (p/2)ln(N)

Where:

- **L** — likelihood of the model  
- **p** — number of parameters of the model  
- **N** — number of data points  

The goal is to maximize the BIC value:  
higher BIC → better balance between **model fit** and **complexity**.

---

### X-means Workflow

1. **Initialization** — start standard K-means with a small number of clusters (e.g., *k = 2*).  
2. **Expansion** — for each current cluster, attempt to split it into two subclusters by running K-means locally.  
3. **Evaluation (BIC)** — compute the BIC for both models:  
   - the **unsplit (simpler)** model  
   - the **split (more complex)** model  
4. **Decision** —  
   - If **BIC increases**, keep the split.  
   - If **BIC decreases**, discard the split.  
5. Repeat the process until **no split** further improves the BIC.

---

### Process Summary

**Initialization → Expansion → Evaluation (BIC) → Decision**

## Partial Fine-Tuning with CLIP

**Fine-Tuning** is the process of adjusting a **pre-trained model** on a specific dataset so that the model learns features more relevant to a particular application.

**Partial Fine-Tuning** updates only **specific layers or weights** of the model, significantly reducing **computational cost** compared to full fine-tuning.

---

### CLIP (Open Source)

- **C**ontrastive **L**anguage-**I**mage **P**re-training  
- Developed by **OpenAI**  
- **Multimodal** — understands both **text** and **images** simultaneously  

CLIP is trained on **millions of image-caption pairs**, aligning **related images and texts** closely in the embedding space, while **unrelated pairs** are far apart.

# How to Run the Model

### Drive / Setup
Download or access the necessary files from the Google Drive folder:  
[Access Google Drive](https://drive.google.com/drive/folders/1cNUnDcWw_KWFhwOR9JDMLZPxZsjevc6N?usp=drive_link)


---

### Install Dependencies
Install all required Python libraries:

```bash
pip install -r requirements.txt
```

### Prepare (Pre-processing)

```bash
python prepare_dataset.py
```

### Training

```bash
python data_treino.py --csv dados.csv --root "Soil types" --batch 4 --epochs 150 --num-workers 0
```


