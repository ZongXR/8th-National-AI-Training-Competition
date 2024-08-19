# 题目说明

有关于是否患心脏病的一份数据集，考生需要通过机器学习方法训练模型，预测结果。
需要完善两份文件code_filling.txt和result.txt：
1.补全demo.py中的代码，并存储到code_filling.txt中，格式参考“code_filling(示例).txt”，一共有22个空缺，共22分。code_filling.txt需要存放在workspace/questionA/目录下；
2.请继续完成对于heart_unlabeled.csv数据的预测代码编写，将预测结果存储到result.txt。预测代码可参照demo.ipynb中数据预测部分。result.txt需要存放在workspace/questionA/目录下；

# 文件说明
demo.ipynb  示例代码
heart.csv  训练数据集(带有标签)
heart_unlabeled.csv  需考生预测的数据集（未带有标签）
code.filling.txt 需要提交的第一份文件（补全demo.ipynb中的空缺），格式参考“code_filling(示例).txt”
result.txt  对heart_unlabeled.csv预测保存的标签结果,“result(示例).txt”
code_filling(示例).txt 保存格式的示例
result(示例).txt 保存的格式示例（请一定按照示例所示保存result.txt文件）

# 分数评判说明（满分50分）
本题目共50分
第一部分22分，计算code_filling.txt中的分值,每空1分
第二部分28分，计算result.txt的f1-score
如果第二部分f1-score低于60%，则第二部分分数为0，否则第二部分分数=f1-score*28

# 标准答案
下面给出标准答案的位置，每位选手可自行编写对比脚本，参照上述【分数评判说明】进行判分练习：
code_filling.txt 的标准答案文件是data/questionAAnswer/code_filling.txt
result.txt 的标准答案文件是data/questionAAnswer/result.txt


# 数据集字段说明

●Age: 年龄
●Sex: 性别
●ChestPainType: 胸痛类型
●RestingBP: 静息血压
●Cholesterol: 胆固醇
●FastingBS: 空腹血糖
●RestingECG: 静息心电图
●MaxHR: 最大心率
●ExerciseAngina: 运动引起的心绞痛
●Oldpeak: 峰值
●ST_Slope: ST段斜率
●HeartDisease: 是否患有心脏病（目标变量）


# 其他说明

建议考试前保存demo.ipynb的副本，以免造成混乱丢失关键信息

仅能使用以下依赖库，且考试环境无法安装其他依赖库（离线环境）

numpy==1.24.4
scikit-learn==1.3.0
matplotlib==3.7.5
transformers==4.38.1
datasets==2.18.0
accelerate==0.26.1
evaluate==0.4.1
bitsandbytes==0.42.0
certifi==2024.6.2
charset-normalizer==3.3.2
colorama==0.4.6
contourpy==1.1.1
cycler==0.12.1
filelock==3.15.4
fonttools==4.53.0
idna==3.7
importlib_resources==6.4.0
Jinja2==3.1.4
kiwisolver==1.4.5
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.1
onnx==1.16.1
opencv-python==4.10.0.84
packaging==23.1
pandas==2.0.3
pillow==9.5.0
protobuf==3.20.3
psutil==6.0.0
py-cpuinfo==9.0.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
requests==2.32.3
scipy==1.10.1
seaborn==0.13.2
setuptools==70.1.1
six==1.16.0
sympy==1.12.1
torch==2.0.0
torchaudio==2.0.1
torchvision==0.15.1
tqdm==4.66.4
typing_extensions==4.12.2
tzdata==2024.1
ultralytics==8.2.45
ultralytics-thop==2.0.0
urllib3==2.2.2
wheel==0.43.0
zipp==3.19.2
modelscope==1.15.0
pycocotools==2.0.8
peft==0.10.0
sentencepiece==0.1.99
streamlit==1.24.0

