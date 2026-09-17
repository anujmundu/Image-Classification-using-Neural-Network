# Contributing Guidelines

Thank you for your interest in contributing to the **11-Architecture Deep Learning Vision Benchmark Suite**!

We welcome contributions from the community to expand model architectures, optimize inference engines, improve visual explainability tools, or add new benchmark datasets.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/anujmundu/Image-Classification-using-Neural-Network.git
   cd Image-Classification-using-Neural-Network
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Test Suite:**
   ```bash
   pytest tests/ -v
   ```

---

## 📐 Coding & Architectural Standards

- **Code Style:** Follow PEP 8 style conventions with descriptive variable names and explicit type annotations.
- **Model Architecture Consistency:** Any new backbone added to `src/model/backbones.py` must:
  1. Accept `num_classes` parameter.
  2. Implement standard NCHW tensor input format `(B, 3, 224, 224)`.
  3. Include unit test coverage in `tests/test_backbones.py`.
- **Reproducibility:** Fix random seeds using `src/train.py` seed controls.
- **AMP Acceleration:** Always verify mixed-precision (`torch.amp.autocast`) compatibility.

---

## 🧪 Testing Guidelines

Before submitting a Pull Request, verify that all unit and integration tests pass:
```bash
pytest tests/ -v
```

---

## 📬 Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-new-backbone
   ```
2. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(backbones): add CoAtNet architecture with unit tests"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/my-new-backbone
   ```
4. Open a Pull Request on GitHub describing your changes, benchmarks, and test results.
