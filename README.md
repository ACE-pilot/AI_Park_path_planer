# AI_Park_path_planer
Automated Path Planning Method for Urban Street-Corner Parks Based on Multi-Agent Deep Reinforcement Learning
![Enscape_2025-02-14-15-17-08](https://github.com/user-attachments/assets/79f7d7d5-5ce1-4a59-bbb4-bbd2880dcfc3)



Hello, and welcome to my code repository. 


## Environment Requirements

To ensure smooth execution and reproducibility of the experiments, please set up your Python environment as follows:

### 1. Python Version

This project is compatible with **Python 3.8+**.

### 2. Core Dependencies

You may install the necessary packages via `pip`:

```bash
pip install -r requirements.txt
```

The key libraries required include:

* **Reinforcement Learning & Environment**:
  `gym==0.26.2`, `parl==2.2.1`

* **Deep Learning & Math**:
  `paddlepaddle-gpu==2.5.1`, `numpy==1.26.2`, `scipy==1.11.4`, `scikit-learn==0.24.2`, `statsmodels==0.14.4`

* **Visualization & Plotting**:
  `matplotlib==3.9.4`, `seaborn==0.13.2`, `tensorboard==2.11.0`, `tensorboardX==2.5`

* **3D Model Visualization (optional)**:
  To visualize the 3D reconstruction results, please install:

  * **Rhino 8**
  * **Grasshopper with required plugins**
  * **Enscape rendering plugin**

### 3. Instructions for running the code

1. You can test the performance of the currently trained model by running the following command in the terminal:

   ```
   python test_CSV.py --restore --show
   ```

   The automatically saved results can be found in the `savegoodresults` folder.
   Please ensure that the model files in the `model` folder have not been overwritten by new training. If they have been overwritten, you can copy the backup from the `z_protect` folder. You can also find the previous `train_log` in the `z_protect` folder.

2. To train a new agent model, simply run:

   ```
   python train.py
   ```
To ensure the authenticity of the research, we have preserved all modified versions of the code. Some older versions may contain errors or conflicts related to library dependencies. If you wish to visualize the 3D models, please install Rhino 8 along with the necessary Grasshopper plugins and the Enscape rendering plugin.
