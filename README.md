# AI_Park_path_planer
Automated Path Planning Method for Urban Street-Corner Parks Based on Multi-Agent Deep Reinforcement Learning

Hello, and welcome to my code repository. Below are the instructions for running the code:

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
