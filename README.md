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

<p align="center">
  <sub>Reconstruction quality as a complementary signal for robust face anti-spoofing.</sub>
</p>
