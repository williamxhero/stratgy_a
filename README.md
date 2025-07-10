# Python项目

这是一个使用Python和虚拟环境的项目模板。

## 项目结构

```
stratgy_a_proj/
├── venv/                 # Python虚拟环境
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
└── README.md           # 项目说明
```

## 环境设置

### 1. 激活虚拟环境

在Windows Git Bash中：
```bash
source venv/Scripts/activate
```

在Windows命令提示符中：
```cmd
venv\Scripts\activate
```

在Linux/Mac中：
```bash
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行程序

```bash
python main.py
```

## 开发指南

1. 在`requirements.txt`中添加项目所需的依赖包
2. 使用`pip freeze > requirements.txt`更新依赖列表
3. 在`main.py`中编写主要逻辑

## 退出虚拟环境

```bash
deactivate
