# Classification_Pytorch

主要目的是培养工程代码实现的可读性

项目文件架构：<br/>
checkpoints/
##data/<br/>
	__init__.py<br/>
	dataset.py<br/>
##models/<br/>
	__init__.py<br/>
	DarkNet.py<br/>
	ResNet34.py<br/>
	...<br/>
##utils/<br/>
	__init__.py<br/>
	visualize.py<br/>
	dataname_change.py<br/>
	...<br/>
##config.py<br/>
##main.py<br/>
##class_demo.py<br/>
##requirements.txt<br/>
##README.md<br/>



- checkpoints/:用于保存训练好的模型
- data/:数据的加载，预处理，实现最终生成喂入网络的格式
- modes/:定义多模型
- utils/:定义一些辅助功能的函数，比如可视化，demo等
- config.py:整个工程课配置的变量都集中于此，并给与初始参数
- main.py:主文件，实现train/test等主要的程序入口
- class_demo.py:将训练好的分类模型应用
- requirements.txt:工程依赖的第三方库申明
- README.md:提供工程的必要说明


##########################<br/>
本项目的功能是将胎儿超声的检测的七个部位进行分类：头部、颜面部、腹部、四肢、心脏、脊柱、胸部

