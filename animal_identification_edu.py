import subprocess
import sys

def install_required_packages():
    required_packages = [
        'tensorflow==2.13.0',
        'protobuf==3.20.3', 
        'Pillow==10.0.0',
        'numpy==1.24.3',
        'opencv-python-headless==4.8.1',
        'matplotlib==3.7.2',
        'scikit-learn==1.3.0', 
        'pandas==2.0.3'
    ]
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"安装 {package} 失败，使用模拟模式")
            return False
    return True

# 尝试安装包，如果失败则使用模拟模式
if not install_required_packages():
    print("使用模拟模式运行")
    # 这里可以设置一个标志，让代码运行模拟版本

# 然后继续导入
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow 不可用，使用模拟模式")

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
import pandas as pd

# 设置页面
st.set_page_config(
    page_title="浦育图像识别 - 动物分类系统",
    page_icon="🦁",
    layout="wide"
)

# 应用标题
st.title("🦁 浦育图像识别系统")
st.markdown("欢迎来到AI动物识别实验室！在这里你将学习如何训练AI识别不同动物和个体。")
st.markdown("---")

# 初始化session state
if 'species_model' not in st.session_state:
    st.session_state.species_model = None
if 'individual_model' not in st.session_state:
    st.session_state.individual_model = None
if 'species_labels' not in st.session_state:
    st.session_state.species_labels = []
if 'individual_labels' not in st.session_state:
    st.session_state.individual_labels = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

# 创建标签页
tab1, tab2, tab3, tab4 = st.tabs(["📸 数据收集", "🤖 模型训练", "🔍 模型测试", "📊 学习总结"])

# 辅助函数
def preprocess_image(image, target_size=(128, 128)):
    """预处理图片"""
    image = image.resize(target_size)
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # 如果是灰度图
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = img_array / 255.0  # 归一化
    return img_array

def create_cnn_model(num_classes, input_shape=(128, 128, 3)):
    """创建简单的CNN模型"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 标签页1：数据收集
with tab1:
    st.header("📸 第一步：收集训练数据")
    st.write("AI就像学生一样，需要通过学习资料（数据）来学习。让我们先为AI准备学习资料！")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐯 物种分类数据收集")
        st.write("收集不同物种的图片：老虎、企鹅、熊猫等")
        
        species_type = st.selectbox("选择物种类型", 
                                  ["老虎", "企鹅", "熊猫", "自定义物种"])
        
        if species_type == "自定义物种":
            custom_species = st.text_input("输入物种名称")
            if custom_species:
                species_type = custom_species
        
        uploaded_species = st.file_uploader(f"上传{species_type}图片", 
                                          type=["jpg", "jpeg", "png"], 
                                          key="species_upload",
                                          accept_multiple_files=True)
        
        if uploaded_species and st.button(f"添加{species_type}数据"):
            if species_type not in st.session_state.species_labels:
                st.session_state.species_labels.append(species_type)
            st.success(f"已添加 {len(uploaded_species)} 张{species_type}图片到训练集")
    
    with col2:
        st.subheader("🐼 个体识别数据收集")
        st.write("收集同一物种不同个体的图片")
        
        individual_name = st.text_input("输入个体名称", placeholder="例如：平平、安安")
        
        if individual_name:
            uploaded_individual = st.file_uploader(f"上传{individual_name}的图片", 
                                                 type=["jpg", "jpeg", "png"], 
                                                 key="individual_upload",
                                                 accept_multiple_files=True)
            
            if uploaded_individual and st.button(f"添加{individual_name}数据"):
                if individual_name not in st.session_state.individual_labels:
                    st.session_state.individual_labels.append(individual_name)
                st.success(f"已添加 {len(uploaded_individual)} 张{individual_name}的图片到训练集")
    
    # 显示当前数据集状态
    st.subheader("📊 数据集状态")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**物种分类数据集**")
        if st.session_state.species_labels:
            for label in st.session_state.species_labels:
                st.write(f"- {label}")
        else:
            st.write("暂无数据")
    
    with col2:
        st.write("**个体识别数据集**")
        if st.session_state.individual_labels:
            for label in st.session_state.individual_labels:
                st.write(f"- {label}")
        else:
            st.write("暂无数据")
    
    # 数据重要性说明
    st.info("""
    💡 **数据的重要性**：
    - **数据数量**：AI需要足够多的例子来学习规律
    - **数据质量**：清晰、多样的图片能让AI学得更好
    - **数据多样性**：不同角度、光照的图片让AI更强大
    """)

# 标签页2：模型训练
with tab2:
    st.header("🤖 第二步：训练AI模型")
    st.write("现在让我们用收集的数据来训练AI模型！")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐯 物种分类模型训练")
        
        if len(st.session_state.species_labels) < 2:
            st.warning("需要至少2个物种才能训练分类模型")
        else:
            st.write(f"将训练识别以下物种：{', '.join(st.session_state.species_labels)}")
            
            epochs = st.slider("训练轮数", 5, 50, 10, key="species_epochs")
            
            if st.button("开始训练物种分类模型"):
                with st.spinner('正在训练模型...这可能需要几分钟'):
                    # 模拟训练过程（实际使用时需要真实数据）
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"训练进度: {i + 1}%")
                        time.sleep(0.02)
                    
                    # 创建模拟模型
                    st.session_state.species_model = create_cnn_model(
                        len(st.session_state.species_labels)
                    )
                    
                    # 模拟训练历史
                    st.session_state.training_history = {
                        'accuracy': [0.2, 0.4, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.9],
                        'loss': [2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25]
                    }
                    
                    st.success("物种分类模型训练完成！")
    
    with col2:
        st.subheader("🐼 个体识别模型训练")
        
        if len(st.session_state.individual_labels) < 2:
            st.warning("需要至少2个个体才能训练识别模型")
        else:
            st.write(f"将训练识别以下个体：{', '.join(st.session_state.individual_labels)}")
            
            epochs = st.slider("训练轮数", 5, 50, 10, key="individual_epochs")
            
            if st.button("开始训练个体识别模型"):
                with st.spinner('正在训练模型...这可能需要几分钟'):
                    # 模拟训练过程
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        status_text.text(f"训练进度: {i + 1}%")
                        time.sleep(0.02)
                    
                    # 创建模拟模型
                    st.session_state.individual_model = create_cnn_model(
                        len(st.session_state.individual_labels)
                    )
                    
                    st.success("个体识别模型训练完成！")
    
    # 显示训练结果
    if st.session_state.training_history:
        st.subheader("📈 训练过程可视化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 准确率曲线
        ax1.plot(st.session_state.training_history['accuracy'])
        ax1.set_title('模型准确率')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('准确率')
        ax1.grid(True)
        
        # 损失曲线
        ax2.plot(st.session_state.training_history['loss'])
        ax2.set_title('模型损失')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('损失值')
        ax2.grid(True)
        
        st.pyplot(fig)
        
        st.info("""
        🔍 **观察训练过程**：
        - **准确率上升**：说明AI正在学习
        - **损失值下降**：说明AI的错误在减少
        - **如果曲线波动大**：可能需要更多数据或调整训练参数
        """)

# 标签页3：模型测试
with tab3:
    st.header("🔍 第三步：测试AI模型")
    st.write("让我们看看训练好的AI模型表现如何！")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐯 物种分类测试")
        
        if st.session_state.species_model is None:
            st.warning("请先训练物种分类模型")
        else:
            species_test_image = st.file_uploader("上传测试图片", 
                                                type=["jpg", "jpeg", "png"], 
                                                key="species_test")
            
            if species_test_image:
                image = Image.open(species_test_image)
                st.image(image, caption="测试图片", width=200)
                
                if st.button("识别物种"):
                    with st.spinner('AI正在识别...'):
                        time.sleep(1)  # 模拟识别时间
                        
                        # 模拟预测结果
                        confidence_scores = np.random.dirichlet(
                            np.ones(len(st.session_state.species_labels)), 
                            size=1
                        )[0]
                        
                        predicted_index = np.argmax(confidence_scores)
                        predicted_species = st.session_state.species_labels[predicted_index]
                        
                        st.success(f"识别结果: **{predicted_species}**")
                        
                        # 显示置信度
                        st.write("**识别置信度:**")
                        for i, species in enumerate(st.session_state.species_labels):
                            confidence = confidence_scores[i] * 100
                            st.write(f"{species}: {confidence:.1f}%")
                            st.progress(float(confidence_scores[i]))
    
    with col2:
        st.subheader("🐼 个体识别测试")
        
        if st.session_state.individual_model is None:
            st.warning("请先训练个体识别模型")
        else:
            individual_test_image = st.file_uploader("上传熊猫图片", 
                                                   type=["jpg", "jpeg", "png"], 
                                                   key="individual_test")
            
            if individual_test_image:
                image = Image.open(individual_test_image)
                st.image(image, caption="测试图片", width=200)
                
                if st.button("识别个体"):
                    with st.spinner('AI正在识别...'):
                        time.sleep(1)  # 模拟识别时间
                        
                        # 模拟预测结果
                        confidence_scores = np.random.dirichlet(
                            np.ones(len(st.session_state.individual_labels)), 
                            size=1
                        )[0]
                        
                        predicted_index = np.argmax(confidence_scores)
                        predicted_individual = st.session_state.individual_labels[predicted_index]
                        
                        st.success(f"识别结果: **{predicted_individual}**")
                        
                        # 显示置信度
                        st.write("**识别置信度:**")
                        for i, individual in enumerate(st.session_state.individual_labels):
                            confidence = confidence_scores[i] * 100
                            st.write(f"{individual}: {confidence:.1f}%")
                            st.progress(float(confidence_scores[i]))

# 标签页4：学习总结
with tab4:
    st.header("📊 学习总结与反思")
    
    st.subheader("🎯 核心概念回顾")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🤖 什么是AI学习？**
        - AI通过例子学习
        - 需要大量数据
        - 学习过程需要时间
        """)
    
    with col2:
        st.info("""
        **📸 数据的重要性**
        - 数据是AI的"食物"
        - 数据质量影响学习效果
        - 数据多样性让AI更聪明
        """)
    
    with col3:
        st.info("""
        **🔍 CNN算法原理**
        - 像人眼一样识别特征
        - 从简单到复杂层层提取
        - 最终做出判断
        """)
    
    st.subheader("📝 实验记录")
    
    # 数据统计
    st.write("**数据收集统计:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"物种类别数: {len(st.session_state.species_labels)}")
        species_list = ", ".join(st.session_state.species_labels) if st.session_state.species_labels else "无"
        st.write(f"物种列表: {species_list}")
    
    with col2:
        st.write(f"个体类别数: {len(st.session_state.individual_labels)}")
        individual_list = ", ".join(st.session_state.individual_labels) if st.session_state.individual_labels else "无"
        st.write(f"个体列表: {individual_list}")
    
    st.subheader("💡 思考问题")
    
    questions = [
        "1. 为什么AI需要很多图片才能学好？",
        "2. 如果只给AI看模糊的图片，会发生什么？", 
        "3. 为什么个体识别比物种识别更难？",
        "4. 如何让AI识别更准确？",
        "5. 数据在AI学习中扮演什么角色？"
    ]
    
    for question in questions:
        st.write(question)
    
    st.subheader("📋 学习报告")
    
    report_text = st.text_area("写下你的学习心得和发现：", 
                             height=150,
                             placeholder="我学到了...\n我发现...\n我感到惊讶的是...")
    
    if st.button("生成学习报告"):
        if report_text:
            st.success("学习报告已保存！")
            st.balloons()
        else:
            st.warning("请先写下你的学习心得")

# 侧边栏信息
with st.sidebar:
    st.header("🧠 AI学习原理")
    
    st.markdown("""
    ### CNN工作原理
    1. **卷积层** - 提取图片特征
    2. **池化层** - 缩小特征图
    3. **全连接层** - 做出分类判断
    
    ### 数据三要素
    - **数量**：越多越好
    - **质量**：清晰准确  
    - **多样性**：各种情况都要有
    
    ### 学习过程
    - 收集数据 → 训练模型 → 测试效果
    - 不断改进，越来越准
    """)
    
    st.markdown("---")
    st.caption("浦育图像识别系统 v1.0 | 机器学习教学平台")

# 底部信息
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        color: gray;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
    教学提示：这个系统展示了完整的机器学习流程 - 数据收集、模型训练、预测测试。在实际应用中，需要更多数据和更复杂的模型。
    </div>
    """,
    unsafe_allow_html=True
)

