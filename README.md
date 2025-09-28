# Semi-Supervised Learning for CT-Scan Segmentation

This project was carried out as part of the **MVA Master's program at ENS Paris-Saclay**  
in the course taught by **Stéphane Mallat (Collège de France)**, in collaboration with **Raidium**.

It was also conducted in the context of the **ENS Data Challenge** on medical imaging.  
Our team ranked **8th out of 36 participants** on the final private leaderboard:  
🔗 https://challengedata.ens.fr/participants/challenges/165/ranking/public

---

## 🧠 Project Motivation

CT scanners enable precise 3D visualization of internal anatomy, but annotations are often
incomplete and expensive to obtain. Our goal was to build a segmentation system that:

✅ Learns from **partially labeled** images  
✅ Leverages **unlabeled data** with semi-supervised methods  
✅ Remains computationally feasible with limited GPU access

Total dataset: **2000 CT images**  
- **800 partially annotated**
- **1200 unlabeled**

This constraint modeled realistic clinical and industrial scenarios.

---

## 🔹 1. Supervised Baseline (U-Net Variants)

We evaluated architectures used in medical imaging:

- ✅ **Attention U-Net (AttUNet)** – best baseline
- MISSFormer
- UC-TransNet

To handle incomplete labels, we used **Dice Loss only** (without cross-entropy):

$$
\mathcal{L}_\text{Dice} = 1 - \frac{2 \cdot \sum_i P_i T_i}{\sum_i P_i + \sum_i T_i + \epsilon}
$$

**Best supervised result (AttUNet, 30 epochs):**  
🎯 Dice score: **0.27** (public leaderboard)

We also experimented with nnU-Net and DinoV2-based decoders, but compute limits prevented full convergence.

---

## 🔹 2. Semi-Supervised Learning

To exploit the 1200 unlabeled scans, we implemented a **Teacher-Student framework**, inspired by \[Baldeon et al., 2023\]:

### ✅ Main Method: Teacher–Student

1. **Teacher training** on labeled data (800 images)  
2. **Pseudo-label generation** for unlabeled data  
3. **Confidence-based filtering** → keep 800 best pseudo-labels  
4. **Student training** with:
   - 800 labeled + 800 pseudo-labeled images
   - Loss = Dice + L2 consistency term

📉 Score improvement: **0.24 → 0.25**  
This increase was marginal but measurable.

### 🔄 Variants Explored

- Pretrain on pseudo-labels, then fine-tune
- Mean Teacher (EMA updates) — partial implementation

---

## 📊 Results & Training Dynamics

Training was constrained by GPU availability  
(Kaggle P100 GPU → +10h per full run).

Here are the final learning curves for Teacher–Student and Pretraining/Finetuning:

![](figs/SSL_TrainingCurves.png)

We observed:
- Slight gain in Dice during the second phase
- Limited impact due to:
  - Modest unlabeled data volume (1.5× ratio)
  - Imperfect pseudo-label quality
  - Partial ground-truth masks
  - Limited hyperparameter tuning due to compute

---

## ✅ Repository Structure

```text
Sem-Supervised-Segmentation-of-CT-Scans/
├── README.md
├── REPORT.pdf
├── src/
│ ├── models/
│ ├── utils/
│ ├── run.py
│ └── Raidium_Challenge_Data_2025_Exemple_Data_Handling.ipynb
```


---

## 🏅 Competition Context

This project was evaluated as part of the  
**ENS Challenge Data - Medical Segmentation Task**, hosted by Collège de France and Raidium.

✅ **Final ranking:** 8th / 36  
🔗 https://challengedata.ens.fr/participants/challenges/165/ranking/public

---

## 📥 Full Report

The complete written project (LaTeX/PDF) is available in:  
📄 `REPORT.pdf`

It includes:
- Methodology
- Architectures
- Figures and curves
- Critical analysis
- Future extensions

---

## 👥 Contributors

- Guillaume Henon-Just & Emilie Pic 
  Master MVA, ENS Paris-Saclay  
  Course by Stéphane Mallat, Collège de France  
  Collaboration with Raidium

---

## 🎯 Keywords

Medical Imaging · Semi-Supervised Learning · U-Net · Pseudo-Labeling · CT Scans · Dice Loss

---

Feel free to reach out for collaboration or research discussions.
