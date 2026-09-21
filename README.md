Reconstruction Quality-based Face Presentation Attack Detection (RQ-PAD)

Face anti-spoofing is essential for securing face recognition systems against presentation attacks such as printed photos, replay attacks, and physical masks. However, detecting increasingly sophisticated and unseen attacks remains challenging.

RQ-PAD is a generative reconstruction-based framework for robust Face Presentation Attack Detection (PAD). The key idea is to exploit the reconstruction behavior of a face generation model trained exclusively on bona fide faces. Given an input face, the model reconstructs a realistic facial representation regardless of whether the input is genuine or spoofed. While bona fide samples tend to be reconstructed consistently with the original input, spoofed samples can introduce discrepancies between the input and its reconstruction.

RQ-PAD explicitly leverages this reconstruction–input discrepancy as an additional discriminative signal for face anti-spoofing. By combining reconstruction quality with learned classification features, the proposed framework enhances the detection of both known and unseen presentation attacks.

We evaluate RQ-PAD on HQ-WMCA and SiW-mV2 under:

Seen-attack scenarios
Unseen-attack scenarios
Cross-database evaluation

The experimental results demonstrate that reconstruction quality provides a strong complementary signal for face anti-spoofing and that RQ-PAD achieves competitive performance against state-of-the-art methods.

Key Idea

Input Face → Generative Reconstruction → Reconstruction–Input Comparison → PAD Classification

The framework exploits what a face generation model has learned about the distribution of genuine faces: rather than asking the generator to directly classify an attack, we analyze how well the generated reconstruction agrees with the observed input.

Citation

If you use this work, please cite:

@inproceedings{bouzid2026rq,
  title={RQ-PAD: Reconstruction Quality for Robust Face Presentation Attack Detection},
  author={Bouzid, Hamza and L{\'e}zoray, Olivier and Rosenberger, Christophe},
  booktitle={International Conference on Pattern Recognition},
  pages={651--666},
  year={2026},
  organization={Springer}
}
