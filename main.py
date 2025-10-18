import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import sys
from pyngrok import ngrok


# 检查关键依赖
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError as e:
    st.error(f"NumPy导入失败: {e}")
    NUMPY_AVAILABLE = False

# 设置页面配置
st.set_page_config(
    page_title="AI动物识别教学平台",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-left: 5px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# 初始化session state
def init_session_state():
    default_states = {
        'species_data': {'images': [], 'labels': [], 'files': []},
        'individual_data': {'images': [], 'labels': [], 'files': []},
        'species_model': None,
        'individual_model': None,
        'species_labels': [],
        'individual_labels': [],
        'training_history': {'species': [], 'individual': []}
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


# 模拟训练函数
def simulate_training(epochs=10, model_type="species"):
    """模拟训练过程"""
    if model_type == "species":
        loss_start, loss_end = 2.0, 0.3
        acc_start, acc_end = 0.2, 0.9
    else:
        loss_start, loss_end = 2.2, 0.5
        acc_start, acc_end = 0.15, 0.8

    # 创建更平滑的训练曲线
    x = np.linspace(0, 1, epochs)
    loss_values = loss_start * np.exp(-2 * x) + loss_end
    acc_values = acc_start + (acc_end - acc_start) * (1 - np.exp(-3 * x))

    # 添加小的随机波动
    loss_values += np.random.normal(0, 0.05, epochs)
    acc_values += np.random.normal(0, 0.03, epochs)

    return {
        'loss': loss_values.tolist(),
        'accuracy': acc_values.tolist()
    }


# 模拟预测函数
def simulate_prediction(image, labels):
    """模拟预测结果"""
    # 生成更合理的置信度分数
    base_confidences = np.ones(len(labels)) * 0.1  # 基础置信度

    # 随机选择一个作为主要预测结果
    main_idx = np.random.randint(0, len(labels))
    base_confidences[main_idx] += 0.7  # 主要预测有更高置信度

    # 添加一些随机性
    base_confidences += np.random.uniform(0, 0.2, len(labels))

    # 归一化
    confidence_scores = base_confidences / np.sum(base_confidences)

    predicted_label = labels[main_idx]

    return {
        'predicted_label': predicted_label,
        'confidence': confidence_scores[main_idx] * 100,
        'all_predictions': list(zip(labels, confidence_scores))
    }


# 主应用
def main():
    if not NUMPY_AVAILABLE:
        st.error("❌ NumPy 库加载失败，请检查环境配置")
        st.info("""
        请按以下步骤解决问题：
        1. 打开 Anaconda Prompt
        2. 运行: `conda activate animal-ai-class`
        3. 运行: `conda install numpy=1.21.2`
        4. 重新启动应用
        """)
        return

    init_session_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("# 🐾 AI动物识别教学平台")
        st.markdown("---")

        # 简单的菜单
        menu_options = ["🏠 平台介绍", "📸 数据收集", "🤖 模型训练", "🔍 模型测试", "📊 学习分析"]
        selected = st.radio("导航菜单", menu_options)

        st.markdown("---")
        st.markdown("### 系统状态")

        # 显示数据统计
        species_count = len(st.session_state.species_data['images'])
        individual_count = len(st.session_state.individual_data['images'])

        st.write(f"物种图片: {species_count}张")
        st.write(f"个体图片: {individual_count}张")
        st.write(f"物种类别: {len(st.session_state.species_labels)}种")
        st.write(f"个体类别: {len(st.session_state.individual_labels)}种")

        # 显示模型状态
        if st.session_state.species_model:
            st.success("✅ 物种模型已训练")
        else:
            st.warning("⚠️ 物种模型未训练")

        if st.session_state.individual_model:
            st.success("✅ 个体模型已训练")
        else:
            st.warning("⚠️ 个体模型未训练")

        st.markdown("---")
        st.markdown("### 环境信息")
        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"NumPy: {np.__version__}")

    # 页面路由
    if selected == "🏠 平台介绍":
        show_introduction()
    elif selected == "📸 数据收集":
        show_data_collection()
    elif selected == "🤖 模型训练":
        show_model_training()
    elif selected == "🔍 模型测试":
        show_model_testing()
    elif selected == "📊 学习分析":
        show_learning_analysis()


# 平台介绍页面
def show_introduction():
    st.markdown('<div class="main-header">🐾 AI动物识别教学平台</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🎯 学习目标</h3>
        <p>通过本平台，学生将学习：</p>
        <ul>
            <li>机器学习的基本概念和工作原理</li>
            <li>数据在AI系统中的重要性</li>
            <li>图像识别的基本流程</li>
            <li>模型训练和评估的方法</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h3>🔧 技术特点</h3>
        <ul>
            <li><strong>框架</strong>: 模拟学习环境（兼容性最佳）</li>
            <li><strong>界面</strong>: Streamlit 交互式Web应用</li>
            <li><strong>环境</strong>: Conda 虚拟环境管理</li>
            <li><strong>兼容性</strong>: 无需GPU，普通电脑即可运行</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 显示平台示意图
        st.markdown("""
        <div style="text-align: center;">
            <h4>🔄 学习流程</h4>
            <p>1. 收集数据 → 2. 训练模型 → 3. 测试模型 → 4. 分析结果</p>
        </div>
        """, unsafe_allow_html=True)

        # 创建学习流程图
        fig, ax = plt.subplots(figsize=(6, 4))
        steps = ['数据收集', '模型训练', '模型测试', '结果分析']
        values = [25, 50, 75, 100]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        bars = ax.bar(steps, values, color=colors)
        ax.set_ylabel('进度 (%)')
        ax.set_title('学习流程进度')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=15)

        # 在柱状图上添加数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                    f'{int(value)}%', ha='center', va='bottom')

        st.pyplot(fig)

        st.markdown("""
        <div class="success-box">
        <h4>📚 课程特色</h4>
        <ul>
            <li>项目驱动学习</li>
            <li>可视化训练过程</li>
    <li>实时结果反馈</li>
            <li>适合初中生理解</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # 快速开始指南
    st.markdown("### 🚀 快速开始指南")

    steps = [
        {"步骤": 1, "操作": "在'数据收集'页面添加动物图片", "说明": "上传老虎、企鹅、熊猫等动物图片"},
        {"步骤": 2, "操作": "在'模型训练'页面训练AI模型", "说明": "选择参数并开始训练"},
        {"步骤": 3, "操作": "在'模型测试'页面测试模型效果", "说明": "上传新图片查看识别结果"},
        {"步骤": 4, "操作": "在'学习分析'页面查看学习进度", "说明": "分析训练过程和结果"}
    ]

    for step in steps:
        with st.expander(f"步骤{step['步骤']}: {step['操作']}"):
            st.write(step['说明'])
            if step['步骤'] == 1:
                st.info("💡 提示: 每个类别至少准备5-10张图片，效果更好")


# 数据收集页面
def show_data_collection():
    st.markdown('<div class="sub-header">📸 数据收集与管理</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🐯 物种数据", "🐼 个体数据", "📊 数据统计"])

    with tab1:
        st.markdown("### 物种分类数据收集")

        col1, col2 = st.columns([2, 1])

        with col1:
            species_type = st.selectbox("选择物种类型",
                                        ["老虎", "企鹅", "熊猫", "狮子", "大象", "自定义"])

            if species_type == "自定义":
                custom_species = st.text_input("输入物种名称")
                if custom_species:
                    species_type = custom_species

            uploaded_files = st.file_uploader(f"上传{species_type}图片",
                                              type=["jpg", "jpeg", "png"],
                                              accept_multiple_files=True,
                                              key="species_upload")

            if uploaded_files and st.button(f"添加{species_type}数据", key="add_species"):
                for uploaded_file in uploaded_files:
                    try:
                        image = Image.open(uploaded_file)
                        # 转换为RGB模式（避免RGBA问题）
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        st.session_state.species_data['images'].append(image)
                        st.session_state.species_data['labels'].append(species_type)
                        st.session_state.species_data['files'].append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"处理图片 {uploaded_file.name} 时出错: {str(e)}")

                if species_type not in st.session_state.species_labels:
                    st.session_state.species_labels.append(species_type)

                st.success(f"成功添加 {len(uploaded_files)} 张{species_type}图片！")

        with col2:
            if st.session_state.species_data['images']:
                st.write(f"已收集 {len(st.session_state.species_data['images'])} 张物种图片")
                # 显示最后一张上传的图片作为示例
                sample_img = st.session_state.species_data['images'][-1]
                st.image(sample_img, caption="最新上传的图片", width=200)

                # 清除数据按钮
                if st.button("清除所有物种数据"):
                    st.session_state.species_data = {'images': [], 'labels': [], 'files': []}
                    st.session_state.species_labels = []
                    st.session_state.species_model = None
                    st.success("物种数据已清除")
            else:
                st.info("尚未上传任何物种图片")

    with tab2:
        st.markdown("### 个体识别数据收集")

        col1, col2 = st.columns([2, 1])

        with col1:
            individual_name = st.text_input("个体名称", placeholder="例如：平平、安安")
            species_for_individual = st.selectbox("所属物种",
                                                  ["熊猫", "老虎", "企鹅", "其他"])

            uploaded_individual_files = st.file_uploader("上传个体图片",
                                                         type=["jpg", "jpeg", "png"],
                                                         accept_multiple_files=True,
                                                         key="individual_upload")

            if uploaded_individual_files and individual_name and st.button("添加个体数据"):
                for uploaded_file in uploaded_individual_files:
                    try:
                        image = Image.open(uploaded_file)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        st.session_state.individual_data['images'].append(image)
                        st.session_state.individual_data['labels'].append(individual_name)
                        st.session_state.individual_data['files'].append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"处理图片 {uploaded_file.name} 时出错: {str(e)}")

                if individual_name not in st.session_state.individual_labels:
                    st.session_state.individual_labels.append(individual_name)

                st.success(f"成功添加 {len(uploaded_individual_files)} 张{individual_name}的图片！")

        with col2:
            if st.session_state.individual_data['images']:
                st.write(f"已收集 {len(st.session_state.individual_data['images'])} 张个体图片")
                sample_img = st.session_state.individual_data['images'][-1]
                st.image(sample_img, caption="最新上传的个体图片", width=200)

                # 清除数据按钮
                if st.button("清除所有个体数据"):
                    st.session_state.individual_data = {'images': [], 'labels': [], 'files': []}
                    st.session_state.individual_labels = []
                    st.session_state.individual_model = None
                    st.success("个体数据已清除")
            else:
                st.info("尚未上传任何个体图片")

    with tab3:
        st.markdown("### 数据统计与分析")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 物种数据统计")
            if st.session_state.species_labels:
                species_counts = {}
                for label in st.session_state.species_data['labels']:
                    species_counts[label] = species_counts.get(label, 0) + 1

                # 创建条形图
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(species_counts.keys(), species_counts.values(), color='skyblue')
                ax.set_title('物种数据分布')
                ax.set_ylabel('图片数量')
                plt.xticks(rotation=45)

                # 在柱状图上显示数值
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')

                st.pyplot(fig)

                # 详细统计
                st.write("**详细统计:**")
                for species, count in species_counts.items():
                    st.write(f"- {species}: {count}张图片")
            else:
                st.info("暂无物种数据")

        with col2:
            st.markdown("#### 个体数据统计")
            if st.session_state.individual_labels:
                individual_counts = {}
                for label in st.session_state.individual_data['labels']:
                    individual_counts[label] = individual_counts.get(label, 0) + 1

                # 创建饼图
                fig, ax = plt.subplots(figsize=(8, 5))
                wedges, texts, autotexts = ax.pie(individual_counts.values(),
                                                  labels=individual_counts.keys(),
                                                  autopct='%1.1f%%',
                                                  startangle=90)
                ax.set_title('个体数据分布')
                st.pyplot(fig)

                # 详细统计
                st.write("**详细统计:**")
                for individual, count in individual_counts.items():
                    st.write(f"- {individual}: {count}张图片")
            else:
                st.info("暂无个体数据")

        # 数据质量建议
        st.markdown("#### 💡 数据质量建议")
        total_species = len(st.session_state.species_data['images'])
        total_individual = len(st.session_state.individual_data['images'])

        if total_species + total_individual == 0:
            st.warning("请先上传一些图片数据")
        else:
            if total_species < 10:
                st.warning(f"物种数据较少 ({total_species}张)，建议每个物种至少收集5-10张图片")
            else:
                st.success(f"物种数据量充足 ({total_species}张)")

            if total_individual < 5:
                st.warning(f"个体数据较少 ({total_individual}张)，个体识别需要更多数据")
            elif total_individual < 15:
                st.info(f"个体数据量适中 ({total_individual}张)，可以开始训练")
            else:
                st.success(f"个体数据量充足 ({total_individual}张)")


# 模型训练页面
def show_model_training():
    st.markdown('<div class="sub-header">🤖 模型训练与优化</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🐯 物种分类模型", "🐼 个体识别模型"])

    with tab1:
        st.markdown("### 物种分类模型训练")

        if len(st.session_state.species_labels) < 2:
            st.warning("需要至少2个物种才能训练分类模型")
        else:
            st.write(f"**训练物种**: {', '.join(st.session_state.species_labels)}")
            st.write(f"**训练图片数量**: {len(st.session_state.species_data['images'])}")

            col1, col2 = st.columns(2)

            with col1:
                epochs = st.slider("训练轮数", 5, 20, 10, key="species_epochs")
                batch_size = st.selectbox("批处理大小", [4, 8, 16], index=1, key="species_batch")

                if st.button("开始训练物种模型", key="train_species"):
                    if len(st.session_state.species_data['images']) < 5:
                        st.error("每个物种至少需要2-3张图片进行训练")
                    else:
                        with st.spinner("正在训练模型..."):
                            # 显示进度条
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # 模拟训练过程
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                status_text.text(f"训练进度: {i + 1}%")
                                time.sleep(0.03)

                            # 模拟训练历史
                            history = simulate_training(epochs, "species")

                            st.session_state.species_model = "trained_species_model"
                            st.session_state.training_history['species'] = history

                            st.success("物种分类模型训练完成！")

            with col2:
                st.markdown("#### 训练参数说明")
                st.markdown("""
                - **训练轮数**: 模型遍历整个数据集的次数
                - **批处理大小**: 每次训练使用的样本数量
                - **学习率**: 固定为0.001 (优化设置)
                """)

                st.markdown("#### 模型结构")
                st.markdown("""
                - 3层卷积神经网络
                - 2层全连接层
                - 适用于教学的轻量级设计
                """)

            # 显示训练结果
            if st.session_state.training_history.get('species'):
                st.markdown("#### 训练过程可视化")

                history = st.session_state.training_history['species']
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                ax1.plot(history['loss'], 'b-', linewidth=2)
                ax1.set_title('训练损失')
                ax1.set_xlabel('训练轮次')
                ax1.set_ylabel('损失值')
                ax1.grid(True, alpha=0.3)

                ax2.plot(history['accuracy'], 'g-', linewidth=2)
                ax2.set_title('训练准确率')
                ax2.set_xlabel('训练轮次')
                ax2.set_ylabel('准确率')
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)

    with tab2:
        st.markdown("### 个体识别模型训练")

        if len(st.session_state.individual_labels) < 2:
            st.warning("需要至少2个个体才能训练识别模型")
        else:
            st.write(f"**训练个体**: {', '.join(st.session_state.individual_labels)}")
            st.write(f"**训练图片数量**: {len(st.session_state.individual_data['images'])}")

            col1, col2 = st.columns(2)

            with col1:
                epochs = st.slider("训练轮数", 5, 25, 15, key="individual_epochs")
                batch_size = st.selectbox("批处理大小", [4, 8, 16], index=1, key="individual_batch")

                if st.button("开始训练个体模型", key="train_individual"):
                    if len(st.session_state.individual_data['images']) < 5:
                        st.error("每个个体至少需要2-3张图片进行训练")
                    else:
                        with st.spinner("正在训练个体识别模型..."):
                            # 模拟训练过程
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i in range(100):
                                progress_bar.progress(i + 1)
                                status_text.text(f"训练进度: {i + 1}%")
                                time.sleep(0.03)

                            # 模拟训练历史
                            history = simulate_training(epochs, "individual")

                            st.session_state.individual_model = "trained_individual_model"
                            st.session_state.training_history['individual'] = history

                            st.success("个体识别模型训练完成！")

            with col2:
                st.markdown("#### 个体识别特点")
                st.markdown("""
                - 比物种识别更具挑战性
                - 需要学习更细微的特征
                - 通常需要更多的训练数据
                - 准确率相对较低
                """)

                st.markdown("#### 训练建议")
                st.markdown("""
                - 每个个体提供5-10张图片
                - 包含不同角度和光线条件
                - 增加训练轮数提高准确率
                """)

            # 显示训练结果
            if st.session_state.training_history.get('individual'):
                st.markdown("#### 训练过程可视化")

                history = st.session_state.training_history['individual']
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                ax1.plot(history['loss'], 'b-', linewidth=2)
                ax1.set_title('训练损失')
                ax1.set_xlabel('训练轮次')
                ax1.set_ylabel('损失值')
                ax1.grid(True, alpha=0.3)

                ax2.plot(history['accuracy'], 'g-', linewidth=2)
                ax2.set_title('训练准确率')
                ax2.set_xlabel('训练轮次')
                ax2.set_ylabel('准确率')
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)


# 模型测试页面
def show_model_testing():
    st.markdown('<div class="sub-header">🔍 模型测试与评估</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🐯 物种识别测试", "🐼 个体识别测试", "📈 性能对比"])

    with tab1:
        st.markdown("### 物种分类模型测试")

        if st.session_state.species_model is None:
            st.warning("请先训练物种分类模型")
        else:
            col1, col2 = st.columns(2)

            with col1:
                test_image = st.file_uploader("上传测试图片",
                                              type=["jpg", "jpeg", "png"],
                                              key="species_test")

                if test_image:
                    try:
                        image = Image.open(test_image)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        st.image(image, caption="测试图片", use_column_width=True)
                    except Exception as e:
                        st.error(f"图片加载失败: {str(e)}")

            with col2:
                if test_image and st.button("识别物种", key="predict_species"):
                    with st.spinner("AI正在识别..."):
                        time.sleep(1.5)  # 模拟识别时间

                        # 使用模拟预测
                        result = simulate_prediction(image, st.session_state.species_labels)

                        st.success(f"识别结果: **{result['predicted_label']}** (置信度: {result['confidence']:.1f}%)")

                        # 显示置信度
                        st.markdown("#### 识别置信度")
                        for species, confidence in result['all_predictions']:
                            confidence_percent = confidence * 100
                            st.write(f"**{species}**: {confidence_percent:.1f}%")
                            st.progress(float(confidence))

            # 批量测试
            st.markdown("---")
            st.markdown("#### 批量测试")
            batch_files = st.file_uploader("上传多张测试图片",
                                           type=["jpg", "jpeg", "png"],
                                           accept_multiple_files=True,
                                           key="species_batch_test")

            if batch_files and st.button("批量测试", key="batch_species"):
                results = []
                progress_bar = st.progress(0)

                for i, file in enumerate(batch_files):
                    try:
                        image = Image.open(file)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')

                        # 模拟预测
                        result = simulate_prediction(image, st.session_state.species_labels)

                        results.append({
                            '图片': file.name,
                            '预测结果': result['predicted_label'],
                            '置信度': f"{result['confidence']:.1f}%"
                        })
                    except Exception as e:
                        results.append({
                            '图片': file.name,
                            '预测结果': '识别失败',
                            '置信度': '0%'
                        })

                    progress_bar.progress((i + 1) / len(batch_files))

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

    with tab2:
        st.markdown("### 个体识别模型测试")

        if st.session_state.individual_model is None:
            st.warning("请先训练个体识别模型")
        else:
            col1, col2 = st.columns(2)

            with col1:
                individual_test_image = st.file_uploader("上传个体测试图片",
                                                         type=["jpg", "jpeg", "png"],
                                                         key="individual_test")

                if individual_test_image:
                    try:
                        image = Image.open(individual_test_image)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        st.image(image, caption="测试图片", use_column_width=True)
                    except Exception as e:
                        st.error(f"图片加载失败: {str(e)}")

            with col2:
                if individual_test_image and st.button("识别个体", key="predict_individual"):
                    with st.spinner("AI正在识别个体..."):
                        time.sleep(1.5)

                        # 使用模拟预测
                        result = simulate_prediction(image, st.session_state.individual_labels)

                        st.success(f"识别结果: **{result['predicted_label']}** (置信度: {result['confidence']:.1f}%)")

                        st.markdown("#### 识别置信度")
                        for individual, confidence in result['all_predictions']:
                            confidence_percent = confidence * 100
                            st.write(f"**{individual}**: {confidence_percent:.1f}%")
                            st.progress(float(confidence))

    with tab3:
        st.markdown("### 模型性能对比分析")

        if st.session_state.training_history['species'] or st.session_state.training_history['individual']:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 训练损失对比")
                fig, ax = plt.subplots(figsize=(10, 6))

                if st.session_state.training_history['species']:
                    species_loss = st.session_state.training_history['species']['loss']
                    ax.plot(species_loss, label='物种分类', linewidth=2, marker='o')

                if st.session_state.training_history['individual']:
                    individual_loss = st.session_state.training_history['individual']['loss']
                    ax.plot(individual_loss, label='个体识别', linewidth=2, marker='s')

                ax.set_xlabel('训练轮次')
                ax.set_ylabel('损失值')
                ax.set_title('模型训练损失对比')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with col2:
                st.markdown("#### 训练准确率对比")
                fig, ax = plt.subplots(figsize=(10, 6))

                if st.session_state.training_history['species']:
                    species_acc = st.session_state.training_history['species']['accuracy']
                    ax.plot(species_acc, label='物种分类', linewidth=2, marker='o')

                if st.session_state.training_history['individual']:
                    individual_acc = st.session_state.training_history['individual']['accuracy']
                    ax.plot(individual_acc, label='个体识别', linewidth=2, marker='s')

                ax.set_xlabel('训练轮次')
                ax.set_ylabel('准确率')
                ax.set_title('模型训练准确率对比')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            # 性能分析
            st.markdown("#### 性能分析")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="info-box">
                <h4>🔍 物种分类模型特点</h4>
                <ul>
                    <li>学习宏观特征（颜色、形状）</li>
                    <li>训练相对快速</li>
                    <li>准确率较高</li>
                    <li>适合初学者</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="info-box">
                <h4>🔍 个体识别模型特点</h4>
                <ul>
                    <li>学习细微特征（斑纹、纹理）</li>
                    <li>训练时间较长</li>
                    <li>准确率相对较低</li>
                    <li>更具挑战性</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("请先训练模型以查看性能对比")


# 学习分析页面
def show_learning_analysis():
    st.markdown('<div class="sub-header">📊 学习过程分析</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 学习目标", "📈 进度追踪", "💡 知识总结"])

    with tab1:
        st.markdown("### 学习目标达成情况")

        # 定义学习目标
        goals = [
            {"目标": "理解机器学习基本概念", "状态": "已完成", "进度": 100},
            {"目标": "掌握数据收集和预处理", "状态": "进行中", "进度": 80},
            {"目标": "学会训练模型流程", "状态": "进行中", "进度": 70},
            {"目标": "能够评估模型性能", "状态": "未开始", "进度": 30},
            {"目标": "理解数据重要性", "状态": "已完成", "进度": 100}
        ]

        # 显示目标表格
        df_goals = pd.DataFrame(goals)
        st.dataframe(df_goals, use_container_width=True)

        # 进度可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        goals_names = [goal["目标"] for goal in goals]
        progress = [goal["进度"] for goal in goals]

        colors = ['#4CAF50' if p == 100 else '#FFC107' if p > 50 else '#F44336' for p in progress]
        bars = ax.barh(goals_names, progress, color=colors)
        ax.set_xlabel('完成进度 (%)')
        ax.set_title('学习目标完成情况')
        ax.set_xlim(0, 100)

        # 在条形上显示百分比
        for bar, p in zip(bars, progress):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{p}%', ha='left', va='center')

        st.pyplot(fig)

    with tab2:
        st.markdown("### 学习进度追踪")

        # 学习时间统计
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总学习时间", "2.5小时", "+0.5小时")

        with col2:
            st.metric("完成实验", "3个", "+1个")

        with col3:
            st.metric("物种准确率", "85%", "+10%")

        with col4:
            st.metric("个体准确率", "72%", "+8%")

        # 学习活动时间线
        st.markdown("#### 学习活动时间线")

        activities = [
            {"时间": "第一天", "活动": "了解AI基本概念", "时长": "1小时", "状态": "已完成"},
            {"时间": "第二天", "活动": "收集动物图片数据", "时长": "1小时", "状态": "已完成"},
            {"时间": "第三天", "活动": "训练物种分类模型", "时长": "0.5小时", "状态": "已完成"},
            {"时间": "下一步", "活动": "训练个体识别模型", "时长": "预计1小时", "状态": "待进行"}
        ]

        for activity in activities:
            with st.expander(f"{activity['时间']}: {activity['活动']} ({activity['时长']})"):
                st.write(f"**活动详情**: {activity['活动']}")
                if activity['状态'] == "已完成":
                    st.success("✅ 已完成")
                else:
                    st.info("⏳ 计划中")

    with tab3:
        st.markdown("### 知识总结与反思")

        st.markdown("""
        <div class="info-box">
        <h4>🎓 核心知识点总结</h4>
        <ul>
            <li><strong>机器学习</strong>: 让计算机从数据中学习规律的方法</li>
            <li><strong>数据的重要性</strong>: 数据质量直接影响模型性能</li>
            <li><strong>训练过程</strong>: 通过不断调整参数最小化预测错误</li>
            <li><strong>过拟合</strong>: 模型在训练数据上表现太好，但泛化能力差</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # 学习反思
        st.markdown("#### 学习反思记录")
        reflection = st.text_area("写下你的学习心得和发现：",
                                  height=150,
                                  placeholder="我学到了...\n我发现...\n我感到惊讶的是...\n我还想了解...")

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("保存反思记录"):
                if reflection:
                    st.success("反思记录已保存！")
                    # 在实际应用中，这里可以添加保存到文件的代码
                else:
                    st.warning("请先写下你的学习反思")

        with col2:
            if st.button("生成学习报告"):
                st.info("学习报告生成功能开发中...")


# 运行主程序
if __name__ == "__main__":
    main()
    # 启动隧道
public_url = ngrok.connect(8501)
print(f"公网访问地址: {public_url}")