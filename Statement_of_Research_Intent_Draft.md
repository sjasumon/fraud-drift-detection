# Statement of Research Intent

**Working title:** Adaptive Fraud Detection Under Concept Drift, Towards Privacy-Preserving Cross-Institution Extension

**Applicant:** Sheikh Joynal Abedin Sumon
**Proposed supervisor:** Professor Yun Sing Koh, School of Computer Science, University of Auckland
**Target programme:** PhD, University of Auckland

---

## 1. Motivation

Financial institutions increasingly rely on machine learning to detect fraudulent transactions in real time, but two persistent problems limit how well these systems work in practice.

First, fraud is not static. Fraudulent behaviour evolves deliberately to evade whatever detection method is currently deployed, meaning a model trained on last year's fraud patterns degrades against this year's — a problem the data mining literature calls **concept drift**. Second, the transaction data most useful for spotting cross-institution fraud patterns (the same actor targeting multiple banks, for instance) is exactly the data that privacy law and competitive concerns prevent institutions from pooling or sharing directly. Any individual bank sees only its own slice of a fraud pattern that may only be visible across several institutions at once.

This proposal addresses both problems in sequence: first building a fraud-detection model that explicitly adapts to concept drift within a single institution's transaction stream, then extending that model so multiple institutions can collaboratively detect cross-institution fraud without exposing raw transaction data to one another.

## 2. Research Questions

1. How can a fraud-detection model be designed to detect and adapt to concept drift in transaction streams without requiring costly full retraining, while maintaining detection accuracy on both established and emerging fraud patterns?
2. Can this drift-adaptive model be extended into a privacy-preserving, multi-institution setting — using techniques such as secure computation or federated learning — without materially sacrificing the detection performance achieved in the single-institution case?
3. What is the practical trade-off between privacy guarantees and detection latency/accuracy in a real-time transaction-monitoring context, and where does that trade-off become acceptable for deployment?

## 3. Proposed Approach

**Phase 1 (Year 1–2): Concept-drift-aware fraud detection, single institution.**
Using publicly available transaction datasets (e.g. IEEE-CIS Fraud Detection, or synthetic transaction generators where real data access isn't available), this phase develops and benchmarks drift-adaptive classification approaches for fraud detection, building on adaptive ensemble and meta-learning techniques from the data stream mining literature. This phase produces a working baseline model and establishes the evaluation methodology (drift detection accuracy, model update cost, and detection performance over time) used throughout the remaining phases.

**Phase 2 (Year 2–3): Privacy-preserving extension.**
The Phase 1 model is extended into a setting where multiple simulated institutions hold private transaction data and wish to jointly improve fraud detection without sharing raw records. This phase investigates secure multi-party computation and/or federated learning approaches for training and updating the drift-adaptive model across institutions, drawing on recent work in secure, GPU-accelerated tree-based model training.

**Phase 3 (Year 3–4): Evaluation, trade-off analysis, and thesis write-up.**
A systematic comparison of detection performance, computational overhead, and privacy guarantees across the single-institution and multi-institution settings, culminating in the thesis.

## 4. Fit with the MARS Lab / Prof. Koh's Research

This proposal sits directly at the intersection of two established strands of Professor Koh's work: her long-standing research in **data stream mining and concept drift detection** (e.g. her work on adaptive ensemble methods for streaming data), and her group's **current work in privacy-preserving machine learning** (e.g. secure decision-tree training on GPUs). It also follows the pattern of several of her past doctoral students, who each anchored a thesis in one well-scoped technical problem within this broader space (fraud detection in online auctions, semi-supervised concept drift detection, meta-learning for stream classification) rather than a broad, unscoped topic.

## 5. Candidate Background

I bring 15+ years of practical, hands-on technical experience in automation, AI systems, and infrastructure — including self-hosted server infrastructure, application auditing, and freelance AI/automation development — alongside a Master of Science in Computer Science (National University of Bangladesh, First Class, including a supervised research thesis). This background gives me a strong practical foundation in building and deploying systems reliably, which I intend to bring to bear on the engineering side of this research alongside the theoretical contributions.

## 6. Next Steps

I would welcome the opportunity to discuss this proposal with you, refine its scope in light of your group's current projects and data access, and explore whether you would be willing to supervise this research as part of a University of Auckland Doctoral Scholarship application (round closing 1 November 2026).

---

*Note: This is a working draft intended to open a conversation with a prospective supervisor, not a final submission document — Professor Koh's feedback is expected to reshape scope, methodology, and framing before formal submission.*
