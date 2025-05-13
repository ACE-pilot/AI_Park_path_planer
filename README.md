您好，欢迎访问我的代码，以下是运行代码的方法：
1.通过在终端运行python test_CSV.py --restore --show来测试当前已经训练好的模型的效果，您可以在savegoodresults文件夹中找到自动保存的结果。请确保model文件夹中的模型文件没有被新的训练覆盖，如果已经被覆盖，可以前往z_protect文件夹中拷贝。您也可以折在z_protect文件夹中看到之前的train_log。
2.通过运行train.py来训练新的智能体模型。
为了体现研究的真实性，我们保留了所有的修改版代码，一些旧版的代码可能在运行库上有冲突或是错误。如果您想尝试可视化三维模型请安装Rhino8及相关grasshopper插件以及Enscape渲染器插件。


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
