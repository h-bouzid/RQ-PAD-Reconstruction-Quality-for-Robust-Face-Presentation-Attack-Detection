# RQ-PAD: Reconstruction Quality for Robust Face Presentation Attack Detection

<p align="center">
  <strong>Generative Reconstruction for Robust Face Anti-Spoofing</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ICPR-2026-blue" alt="ICPR 2026">
  <img src="https://img.shields.io/badge/Face%20PAD-Anti--Spoofing-purple" alt="Face PAD">
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-orange" alt="Deep Learning">
</p>

---

## 🔍 Overview

**Face anti-spoofing (PAD)** is essential for securing face recognition systems against presentation attacks such as **printed photos, replay attacks, and physical masks**. However, detecting increasingly sophisticated and unseen attacks remains challenging for existing approaches.

**RQ-PAD** is a generative reconstruction-based framework for robust **Face Presentation Attack Detection**.

The key idea is to exploit the reconstruction behavior of a face generation model trained **exclusively on bona fide faces**.

Given an input face, the generative model reconstructs a realistic facial representation regardless of whether the input is genuine or spoofed. While bona fide samples tend to produce reconstructions that remain consistent with the original input, spoofed samples can introduce characteristic **discrepancies between the input and its reconstruction**.

RQ-PAD explicitly exploits this reconstruction quality as an additional discriminative signal for face anti-spoofing.

---

## 💡 Key Idea

<p align="center">

**Input Face**
↓
**Generative Face Reconstruction**
↓
**Reconstruction ↔ Input Comparison**
↓
**Reconstruction Quality Features**
↓
**PAD Classification**

</p>

Rather than training the generative model to directly classify attacks, we ask:

> **How well can a model trained only on genuine faces reconstruct the observed input?**

The reconstruction quality provides complementary information that can help distinguish bona fide faces from presentation attacks.

---

## 🧠 Method

RQ-PAD is based on the observation that a face generation model trained exclusively on bona fide facial data learns the characteristics of genuine faces.

For an input image \(x\), the model generates a reconstruction:

$$
\hat{x} = G(x)
$$

The reconstruction is then compared with the original input to obtain reconstruction-quality information:

$$
Q(x,\hat{x}) = D(x,\hat{x})
$$

where \(D(\cdot,\cdot)\) measures the discrepancy between the input and its reconstruction.

These reconstruction-based signals are integrated into the PAD classifier together with learned facial features.

### Why reconstruction quality?

A presentation attack may contain a facial appearance that is visually convincing but does not necessarily correspond to the distribution of genuine facial observations learned by the generative model.

Therefore, the **difference between what is observed and what the genuine-face model reconstructs** can provide an additional cue for detecting attacks.

---

## 📊 Evaluation

We evaluate RQ-PAD on:

| Dataset            | Evaluation                     |
| ------------------ | ------------------------------ |
| **HQ-WMCA**        | Seen attacks                   |
| **HQ-WMCA**        | Unseen attacks                 |
| **SiW-mV2**        | Seen & unseen attacks          |
| **Cross-database** | Generalization across datasets |

The experiments demonstrate that **reconstruction quality provides a strong complementary signal for face anti-spoofing**, improving robustness against both known and previously unseen presentation attacks.

---

## 🏆 Results

RQ-PAD is evaluated against state-of-the-art face anti-spoofing methods across multiple evaluation protocols, including **seen attacks, unseen attacks, and cross-database scenarios**.

> Detailed quantitative results, experimental protocols, and comparisons with state-of-the-art methods are provided in the paper.

---

## 📄 Publication

**RQ-PAD: Reconstruction Quality for Robust Face Presentation Attack Detection**

**Hamza Bouzid**, Olivier Lezoray, Christophe Rosenberger

*International Conference on Pattern Recognition (ICPR), 2026*

📖 **Pages:** 651–666

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{bouzid2026rq,
  title={RQ-PAD: Reconstruction Quality for Robust Face Presentation Attack Detection},
  author={Bouzid, Hamza and L{\'e}zoray, Olivier and Rosenberger, Christophe},
  booktitle={International Conference on Pattern Recognition},
  pages={651--666},
  year={2026},
  organization={Springer}
}
```

---

## 👥 Authors

**Hamza Bouzid** · **Olivier Lezoray** · **Christophe Rosenberger**

GREYC — CNRS UMR 6072
Université de Caen Normandie
ENSICAEN

---

<p align="center">
  <sub>Reconstruction quality as a complementary signal for robust face anti-spoofing.</sub>
</p>
