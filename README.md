# V-JEPA 2 Fine-tuning Comparison: Parameter-Efficient Approaches

This repository demonstrates and compares two parameter-efficient fine-tuning approaches for Meta's V-JEPA 2 (1.2B parameter video embedding model) on a subset of the UCF-101 action recognition dataset.

## 🎯 **Approaches Compared**

### 1. **Final Layer Only Fine-tuning**
- **Strategy**: Freeze the entire V-JEPA 2 backbone, train only the classification head
- **Trainable Parameters**: 10,240 (0.003% of total model)
- **Use Case**: Maximum parameter efficiency, minimal computational overhead

### 2. **LoRA + Final Layer Fine-tuning**
- **Strategy**: Freeze V-JEPA 2 backbone, add LoRA adapters to attention modules + train classification head
- **Trainable Parameters**: 501,770 (0.13% of total model)
- **Use Case**: Better performance with modular, swappable adapters

## 📊 **Key Results**

| Approach | Test Accuracy | Trainable Params | Training Time | Parameter Efficiency |
|----------|---------------|------------------|---------------|---------------------|
| **Final Layer Only** | 97.33% | 10K | 31.5 min | 95.05 acc/M params |
| **LoRA + Final Layer** | **100.00%** | 501K | 31.4 min | 1.99 acc/M params |

**Key Finding**: LoRA achieves perfect accuracy (+2.7% improvement) with 49x more parameters but similar training time.

## 📁 **Repository Structure**

```
vjepa2/
├── 📄 README.md                                    # This file
├── 📄 requirements.txt                             # Python dependencies
│
├── 🧠 SOURCE CODE
│   └── src/
│       ├── common/                                # Shared modules
│       │   ├── utils.py                          # Shared utilities (seed, device, parameter counting)
│       │   ├── data.py                           # Dataset classes, data loading, transforms
│       │   └── training.py                       # Evaluation functions, TensorBoard setup
│       ├── vjepa2_finetuning.py                  # Final layer only fine-tuning
│       └── lora_vjepa2_finetuning.py             # LoRA + final layer fine-tuning
│
├── 🔧 ANALYSIS SCRIPTS
│   └── scripts/
│       ├── compare_results.py                     # Comprehensive comparison analysis script
│       ├── compare_lora_videos.py                 # Video count comparison analysis
│       ├── create_comparison_plot.py              # Create training comparison plots
│       └── plots_as_function_of_fpc.py            # Frames per clip analysis
│
├── 📊 RESULTS & OUTPUTS
│   └── results/
│       ├── metrics/                               # Training metrics JSONs
│       │   ├── lora_training_metrics_*.json      # Various training configurations
│       │   └── lora_training_metrics.json        # Main LoRA metrics
│       ├── *.png                                  # All visualization plots
│       └── *.csv                                  # Comparison tables and analysis
│
├── 📝 TRAINING LOGS
│   └── logs/
│       ├── final_layer_training_output_refactored.txt # Complete training log (final layer)
│       └── lora_training_output_refactored.txt    # Complete training log (LoRA)
│
├── 📚 DOCUMENTATION
│   └── docs/
│       ├── lora_architecture_diagrams.md          # LoRA architecture documentation
│       └── SYS_DESIGN.md                          # System design documentation
│
├── 📦 DATA & DEPLOYMENT
│   ├── UCF101_subset/                             # Dataset (train/val/test splits)
│   └── vjepa2-cloud-run/                          # Cloud deployment configuration
└── 🔧 CONFIGURATION
    ├── .gitignore                                 # Git ignore patterns
    └── .venv/                                     # Python virtual environment
```

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Clone repository
git clone git@github.com:ZacCarrico/vjepa2.git
cd vjepa2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run Training Experiments**
```bash
# Run final layer only fine-tuning
python src/vjepa2_finetuning.py

# Run LoRA + final layer fine-tuning
python src/lora_vjepa2_finetuning.py

# Generate comparison analysis
python scripts/compare_results.py
```

### 3. **View Results**
- **Training curves**: `results/training_comparison.png`
- **Parameter efficiency**: `results/efficiency_analysis.png`
- **Detailed metrics**: `results/comparison_table.csv`
- **Training logs**: `logs/`
- **TensorBoard logs**: `tensorboard --logdir runs`

## 🎛️ **Configuration Options**

### **Training Parameters**
- **Epochs**: 5 (configurable in training scripts)
- **Batch Size**: 1 (due to memory constraints)
- **Learning Rate**: 1e-5 (final layer), 2e-4 (LoRA)
- **Gradient Accumulation**: 4 steps
- **Seed**: 1 (for reproducibility)

### **LoRA Configuration**
- **Rank**: 16
- **Alpha**: 32.0
- **Dropout**: 0.1
- **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `out_proj` (attention layers)

### **Model Details**
- **Base Model**: `facebook/vjepa2-vitl-fpc16-256-ssv2`
- **Total Parameters**: ~375M
- **Backbone**: Frozen (both approaches)
- **Device**: Auto-detected (CUDA > MPS > CPU)

## 📊 **Dataset Information**

### **UCF-101 Subset**
- **Classes**: 10 (subset of original 101 classes)
- **Total Videos**: 405
- **Splits**: train/val/test
- **Video Format**: .avi files
- **Action Categories**: BalanceBeam, BasketballDunk, BandMarching, etc.

**Why Subset?** Using 10 classes instead of full 101 classes enables:
- Faster training and experimentation
- Easier achievement of high accuracy for demonstration
- Reduced computational requirements
- Clear comparison of fine-tuning approaches

## 🔍 **Technical Implementation**

### **Code Architecture**
- **Modular Design**: Common functionality extracted into `common/` modules
- **~60% Code Deduplication**: Shared utilities, data loading, and training functions
- **Consistent Interface**: Both approaches use identical data loading and evaluation pipelines
- **Reproducible**: Fixed seeds and deterministic training

### **LoRA Implementation**
- **Custom LoRA Layer**: Implements Low-Rank Adaptation from scratch
- **Adaptive Integration**: Seamlessly integrates with V-JEPA 2 architecture
- **Parameter Efficiency**: Only 0.13% of model parameters are trainable
- **Modular Adapters**: LoRA weights can be saved/loaded independently

### **Performance Monitoring**
- **Real-time Logging**: Training loss, validation accuracy, timing metrics
- **TensorBoard Integration**: Visual training progress monitoring
- **Comprehensive Analysis**: Automated comparison report generation
- **Parameter Tracking**: Detailed parameter count and efficiency metrics

## 📈 **Key Insights**

1. **Perfect Accuracy**: LoRA achieved 100% test accuracy on the 10-class subset
2. **Parameter Trade-off**: 49x more parameters for 2.7% accuracy improvement
3. **Training Efficiency**: Both approaches have similar training times (~31.5 minutes)
4. **Parameter Efficiency**: Final layer approach is 47.7x more parameter-efficient
5. **Scalability**: LoRA adapters can be easily swapped for different tasks

## 🎯 **Recommendations**

### **Use Final Layer Only When:**
- Maximum parameter efficiency is crucial
- Limited computational resources
- Simple adaptation tasks
- Storage/memory constraints are primary concerns

### **Use LoRA + Final Layer When:**
- Best possible performance is required
- Modular task-specific adapters are desired
- Multiple task adaptation scenarios
- Slight parameter increase is acceptable for accuracy gains

## 🛠️ **Future Extensions**

- **Full UCF-101**: Scale to complete 101-class dataset
- **Other Adapters**: Implement Prefix-tuning, Prompt-tuning variants
- **Model Comparison**: Test different V-JEPA 2 model sizes
- **Advanced LoRA**: Experiment with different rank/alpha configurations
- **Multi-dataset**: Evaluate across different video classification datasets

## 📚 **References**

- **V-JEPA 2**: [Meta's V-JEPA 2 Model](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6)
- **LoRA Paper**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **UCF-101 Dataset**: [UCF-101 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
- **Original Notebook**: [V-JEPA 2 Fine-tuning Notebook](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/notebook_finetuning.ipynb)

## 💻 **Hardware Used**

- **Tested On**: MacBook Pro M3
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ for dataset and model weights
- **Compute**: GPU recommended but not required (MPS/CUDA auto-detected)

## 📄 **License**

This project is for educational and research purposes. Please refer to the respective model and dataset licenses for commercial usage.
