"""
Streamlit интерфейс для подбора терразитовой штукатурки
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import base64

# Настройка страницы
st.set_page_config(
    page_title="Terrazite AI - Подбор штукатурки",
    page_icon="🏗️",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1.5rem;
    }
    .recipe-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .component-bar {
        background-color: #3498db;
        height: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">🏗️ Terrazite AI - Подбор терразитовой штукатурки</h1>', unsafe_allow_html=True)

# Настройки API
API_URL = st.sidebar.text_input("URL API", "http://localhost:8000")

# Основные вкладки
tab1, tab2, tab3 = st.tabs(["🔍 Анализ изображения", "📊 База рецептов", "📝 Добавить новый рецепт"])

with tab1:
    st.markdown('<h2 class="sub-header">Загрузите фото терразита для анализа</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Загрузка изображения
        uploaded_file = st.file_uploader("Выберите изображение", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Показать изображение
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Кнопка анализа
            if st.button("🔬 Проанализировать рецепт", type="primary"):
                with st.spinner("Анализ изображения..."):
                    try:
                        # Отправка на API
                        files = {"image": uploaded_file.getvalue()}
                        response = requests.post(f"{API_URL}/api/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Анализ завершен!")
                            
                            # Отображение результатов
                            st.markdown(f"**Тип заполнителя:** {result['aggregate_type']}")
                            st.markdown(f"**Уверенность:** {result['confidence']:.1f}%")
                            st.markdown(f"**Время обработки:** {result['processing_time_ms']:.0f} мс")
                            
                            # Визуализация компонентов
                            if result['components']:
                                st.markdown("### 📊 Состав рецепта:")
                                
                                # Создаем DataFrame для визуализации
                                df = pd.DataFrame(result['components'])
                                df = df.sort_values('weight_kg', ascending=False)
                                
                                # Столбчатая диаграмма
                                fig = px.bar(
                                    df, 
                                    x='name', 
                                    y='weight_kg',
                                    title="Компоненты рецепта (кг)",
                                    color='weight_kg',
                                    color_continuous_scale='Blues'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Таблица
                                st.dataframe(df, use_container_width=True)
                            
                            # Похожие рецепты
                            if result.get('similar_recipes'):
                                st.markdown("### 🔍 Похожие рецепты:")
                                for similar in result['similar_recipes']:
                                    with st.expander(f"{similar['name']} (схожесть: {similar.get('similarity_score', 0)*100:.0f}%)"):
                                        st.write(f"ID: {similar['recipe_id']}")
                                        st.write(f"Тип: {similar.get('type', 'неизвестно')}")
                        else:
                            st.error(f"❌ Ошибка API: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
    
    with col2:
        # Примеры изображений
        st.markdown("### 📸 Примеры для тестирования:")
        
        example_images = {
            "Мраморный терразит": "https://via.placeholder.com/400x300/3498db/FFFFFF?text=Мраморный+образец",
            "Кварцевый терразит": "https://via.placeholder.com/400x300/e74c3c/FFFFFF?text=Кварцевый+образец",
            "Гранитный терразит": "https://via.placeholder.com/400x300/2ecc71/FFFFFF?text=Гранитный+образец"
        }
        
        for name, url in example_images.items():
            if st.button(f"Использовать пример: {name}"):
                st.info(f"Загружен пример: {name}")
                st.image(url, caption=name, use_column_width=True)

with tab2:
    st.markdown('<h2 class="sub-header">База рецептов терразитовой штукатурки</h2>', unsafe_allow_html=True)
    
    # Поиск и фильтрация
    col1, col2, col3 = st.columns(3)
    with col1:
        search_query = st.text_input("🔍 Поиск по названию")
    with col2:
        aggregate_filter = st.selectbox("Тип заполнителя", ["Все", "мрамор", "кварц", "гранит", "слюда"])
    with col3:
        sort_by = st.selectbox("Сортировать по", ["названию", "типу", "дате создания"])
    
    # Пример данных (заглушка)
    sample_recipes = [
        {
            "recipe_id": "TER_001",
            "name": "Терразит К62А",
            "type": "терразит",
            "main_aggregate": "мрамор",
            "components": {"Цемент белый": 100, "Песок": 342, "Мрамор": 250},
            "total_weight": 1000
        },
        {
            "recipe_id": "TER_002", 
            "name": "Терразит кварцевый",
            "type": "терразит",
            "main_aggregate": "кварц",
            "components": {"Цемент белый": 150, "Песок": 400, "Кварц": 200},
            "total_weight": 1000
        }
    ]
    
    # Отображение рецептов
    for recipe in sample_recipes:
        if search_query.lower() in recipe['name'].lower() and (aggregate_filter == "Все" or recipe['main_aggregate'] == aggregate_filter):
            with st.container():
                st.markdown(f"""
                <div class="recipe-card">
                    <h3>{recipe['name']}</h3>
                    <p><strong>ID:</strong> {recipe['recipe_id']} | <strong>Тип:</strong> {recipe['type']} | <strong>Заполнитель:</strong> {recipe['main_aggregate']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Детали по клику
                with st.expander("Детали рецепта"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Состав (кг на 1000кг):**")
                        for comp, weight in recipe['components'].items():
                            st.write(f"{comp}: {weight} кг")
                    with col2:
                        # Простая визуализация
                        st.write("**Визуализация:**")
                        for comp, weight in recipe['components'].items():
                            percentage = (weight / recipe['total_weight']) * 100
                            st.write(f"{comp}:")
                            st.progress(percentage / 100)

with tab3:
    st.markdown('<h2 class="sub-header">Добавление нового рецепта в базу</h2>', unsafe_allow_html=True)
    
    with st.form("new_recipe_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            recipe_name = st.text_input("Название рецепта")
            recipe_type = st.selectbox("Тип", ["терразит", "шовный", "декоративный"])
            main_aggregate = st.selectbox("Основной заполнитель", ["мрамор", "кварц", "гранит", "слюда"])
        
        with col2:
            total_weight = st.number_input("Общий вес (кг)", min_value=100, max_value=5000, value=1000)
            image_upload = st.file_uploader("Фото образца", type=['jpg', 'jpeg', 'png'])
        
        st.markdown("### 📊 Компоненты рецепта")
        
        # Динамические поля для компонентов
        components = []
        num_components = st.number_input("Количество компонентов", min_value=1, max_value=20, value=5)
        
        for i in range(num_components):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                name = st.text_input(f"Название компонента {i+1}", key=f"comp_name_{i}")
            with col2:
                weight = st.number_input(f"Вес (кг)", min_value=0.0, key=f"comp_weight_{i}")
            with col3:
                unit = st.selectbox(f"Ед.", ["кг", "%"], key=f"comp_unit_{i}")
            
            if name and weight > 0:
                components.append({"name": name, "weight": weight, "unit": unit})
        
        # Кнопка отправки
        submitted = st.form_submit_button("💾 Сохранить рецепт")
        
        if submitted:
            if recipe_name and components:
                st.success(f"Рецепт '{recipe_name}' сохранен!")
                
                # Показать сводку
                st.markdown("### 📋 Сводка рецепта")
                st.write(f"**Название:** {recipe_name}")
                st.write(f"**Тип:** {recipe_type}")
                st.write(f"**Основной заполнитель:** {main_aggregate}")
                st.write(f"**Общий вес:** {total_weight} кг")
                
                # Таблица компонентов
                df = pd.DataFrame(components)
                st.dataframe(df, use_container_width=True)
            else:
                st.error("Заполните обязательные поля!")

# Боковая панель с информацией
st.sidebar.markdown("## ℹ️ О проекте")
st.sidebar.info("""
**Terrazite AI** - система для автоматического 
подбора рецептов терразитовой штукатурки 
по фотографиям образцов.

Использует компьютерное зрение и 
машинное обучение для анализа текстур.
""")

st.sidebar.markdown("## 🚀 Быстрый старт")
st.sidebar.code("""
# Запуск API сервера
uvicorn src.api.main:app --reload

# Запуск интерфейса
streamlit run streamlit_app.py
""")

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>Terrazite AI © 2024 | Система подбора строительных смесей</p>
</div>
""", unsafe_allow_html=True)
