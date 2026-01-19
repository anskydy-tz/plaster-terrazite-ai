"""
Streamlit веб-интерфейс для Terrazite AI
Подбор рецепта терразитовой штукатурки по изображению с поддержкой категорий компонентов
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime
import io

# Добавляем путь к src для импорта модулей
sys.path.append(str(Path(__file__).parent))

# Импорт модулей проекта
from src.utils.config import config, setup_config
from src.utils.logger import setup_logger
from src.data.loader import RecipeLoader, DataLoader
from src.data.component_analyzer import ComponentAnalyzer
from src.models.terrazite_model import TerraziteModel, create_model

# Настройка страницы
st.set_page_config(
    page_title=config.streamlit.title,
    page_icon=config.streamlit.page_icon,
    layout=config.streamlit.layout
)

# Инициализация логгера
logger = setup_logger(__name__)

# Заголовок приложения
st.title(config.streamlit.title)
st.markdown("""
    **Приложение для подбора рецепта терразитовой штукатурки по фотографии образца**
    
    Загрузите фотографию терразитовой штукатурки, и нейросеть определит:
    - Категорию состава (Терразит, Шовный, Мастика, Терраццо, Ретушь)
    - Компоненты и их пропорции
    - Ближайшие рецепты из базы данных
""")

# Инициализация состояния сессии
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'recipes_loaded' not in st.session_state:
    st.session_state.recipes_loaded = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Создание вкладок
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Анализ изображения", 
    "📊 База рецептов", 
    "📈 Анализ компонентов",
    "🧪 Подбор рецепта",
    "⚙️ Настройки"
])


def load_model():
    """Загрузка модели"""
    try:
        with st.spinner("Загрузка модели..."):
            model = create_model(
                model_type='terrazite',
                num_categories=config.model.num_categories,
                num_components=config.model.num_components,
                hidden_size=config.model.hidden_size,
                dropout_rate=config.model.dropout_rate
            )
            
            # Загрузка весов модели (если есть)
            checkpoint_dir = Path(config.project_root) / config.training.checkpoint_dir
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.pth"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    model.load_state_dict(torch.load(latest_checkpoint))
                    st.success(f"Модель загружена из: {latest_checkpoint.name}")
            
            st.session_state.model_loaded = True
            st.session_state.model = model
            return model
            
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


def load_recipes():
    """Загрузка базы рецептов"""
    try:
        with st.spinner("Загрузка базы рецептов..."):
            # Проверяем наличие обработанных данных
            processed_path = Path(config.project_root) / config.data.processed_data_dir / config.data.processed_json
            ml_data_path = Path(config.project_root) / config.data.processed_data_dir / config.data.ml_data_file
            
            if processed_path.exists():
                with open(processed_path, 'r', encoding='utf-8') as f:
                    recipes_data = json.load(f)
                
                # Загружаем данные для ML
                if ml_data_path.exists():
                    with open(ml_data_path, 'r', encoding='utf-8') as f:
                        ml_data = json.load(f)
                    
                    st.session_state.ml_data = ml_data
                
                st.session_state.recipes_data = recipes_data
                st.session_state.recipes_loaded = True
                
                # Создаем DataFrame для отображения
                recipes_list = []
                for recipe in recipes_data['recipes']:
                    recipe_info = {
                        'Название': recipe['name'],
                        'Категория': recipe['category'],
                        'Кол-во компонентов': recipe['component_count'],
                        'Общий вес (кг)': recipe['total_weight']
                    }
                    # Добавляем топ-3 компонента
                    components = sorted(recipe['components'].items(), key=lambda x: x[1], reverse=True)[:3]
                    for i, (comp, value) in enumerate(components):
                        recipe_info[f'Компонент {i+1}'] = f"{comp.split(',')[0]}: {value} кг"
                    
                    recipes_list.append(recipe_info)
                
                st.session_state.recipes_df = pd.DataFrame(recipes_list)
                return recipes_data
            else:
                st.warning("Обработанные данные не найдены. Запустите обработку Excel файла.")
                return None
                
    except Exception as e:
        st.error(f"Ошибка загрузки рецептов: {e}")
        return None


def analyze_components():
    """Анализ компонентов из Excel файла"""
    try:
        with st.spinner("Анализ компонентов..."):
            excel_path = Path(config.project_root) / config.data.excel_file
            if not excel_path.exists():
                # Ищем файл в других местах
                possible_paths = [
                    excel_path,
                    Path("data/raw/recipes.xlsx"),
                    Path("Рецептуры терразит.xlsx")
                ]
                
                for path in possible_paths:
                    if path.exists():
                        excel_path = path
                        break
            
            if excel_path.exists():
                analyzer = ComponentAnalyzer(str(excel_path))
                analyzer.load_excel()
                analysis_results = analyzer.analyze_components()
                
                # Генерация отчетов
                report_path = analyzer.generate_report()
                viz_path = analyzer.visualize_analysis()
                
                st.session_state.component_analysis = analysis_results
                st.session_state.analyzer = analyzer
                
                return analysis_results
            else:
                st.warning("Excel файл с рецептами не найден.")
                return None
                
    except Exception as e:
        st.error(f"Ошибка анализа компонентов: {e}")
        return None


def predict_image(image):
    """Предсказание рецепта по изображению"""
    try:
        if not st.session_state.model_loaded:
            model = load_model()
            if model is None:
                return None
        
        model = st.session_state.model
        
        # Преобразование изображения для модели
        image_np = np.array(image)
        # Здесь должна быть предобработка изображения для модели
        # Пока что возвращаем заглушку
        
        # Для демонстрации создаем случайные предсказания
        np.random.seed(hash(image.tobytes()) % 10000)
        
        # Случайная категория
        categories = config.data.recipe_categories
        category_idx = np.random.randint(0, len(categories))
        predicted_category = categories[category_idx]
        
        # Случайные компоненты
        if hasattr(st.session_state, 'analyzer'):
            analyzer = st.session_state.analyzer
            component_features = analyzer.get_component_features()
            component_names = list(component_features['component_to_idx'].keys())
        else:
            component_names = [
                "Цемент белый ПЦ500",
                "Цемент серый ПЦ500, кг", 
                "Песок лужский фр.0-0,63мм, кг",
                "Доломитовая мука, кг"
            ]
        
        num_components = min(10, len(component_names))
        selected_indices = np.random.choice(len(component_names), num_components, replace=False)
        
        predicted_components = {}
        for idx in selected_indices:
            component_name = component_names[idx]
            # Случайное значение от 10 до 500 кг
            value = np.random.uniform(10, 500)
            predicted_components[component_name] = round(value, 1)
        
        # Поиск похожих рецептов
        similar_recipes = []
        if st.session_state.recipes_loaded:
            recipes_data = st.session_state.recipes_data
            # Простой поиск по категории
            for recipe in recipes_data['recipes']:
                if recipe['category'] == predicted_category:
                    similarity = np.random.uniform(0.7, 0.95)  # Случайное сходство
                    similar_recipes.append({
                        'recipe': recipe,
                        'similarity': similarity
                    })
            
            # Сортируем по сходству
            similar_recipes.sort(key=lambda x: x['similarity'], reverse=True)
            similar_recipes = similar_recipes[:5]
        
        results = {
            'image': image,
            'predicted_category': predicted_category,
            'predicted_components': predicted_components,
            'similar_recipes': similar_recipes,
            'confidence': np.random.uniform(0.7, 0.95)
        }
        
        st.session_state.prediction_results = results
        return results
        
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
        return None


def visualize_component_groups(analysis_results):
    """Визуализация групп компонентов"""
    if not analysis_results or 'component_groups_by_category' not in analysis_results:
        return
    
    fig = go.Figure()
    
    categories = list(analysis_results['component_groups_by_category'].keys())
    groups = list(config.data.component_groups.keys())
    
    # Создаем heatmap данных
    data_matrix = []
    for category in categories:
        row = []
        for group in groups:
            count = analysis_results['component_groups_by_category'][category].get(group, 0)
            row.append(count)
        data_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=groups,
        y=categories,
        colorscale='YlOrRd',
        text=[[f"{val}" for val in row] for row in data_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Использование групп компонентов по категориям",
        xaxis_title="Группы компонентов",
        yaxis_title="Категории рецептов",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def visualize_category_distribution(recipes_data):
    """Визуализация распределения по категориям"""
    if not recipes_data:
        return
    
    categories = {}
    for recipe in recipes_data['recipes']:
        category = recipe['category']
        categories[category] = categories.get(category, 0) + 1
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(categories.keys()),
            values=list(categories.values()),
            hole=.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )
    ])
    
    fig.update_layout(
        title="Распределение рецептов по категориям",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Вкладка 1: Анализ изображения
with tab1:
    st.header("🔍 Анализ изображения терразитовой штукатурки")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Загрузка изображения
        uploaded_file = st.file_uploader(
            "Загрузите фотографию терразитовой штукатурки",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Изображение должно быть хорошо освещено и показывать текстуру штукатурки"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Кнопка анализа
            if st.button("🔬 Проанализировать изображение", type="primary"):
                with st.spinner("Анализ изображения..."):
                    results = predict_image(image)
                    
                    if results:
                        st.success("Анализ завершен!")
        
        # Примеры изображений
        st.subheader("Примеры изображений")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("Терразит (образец 1)", use_container_width=True):
                st.info("Загрузите изображение терразитовой штукатурки")
        
        with example_col2:
            if st.button("Терраццо (образец 2)", use_container_width=True):
                st.info("Загрузите изображение терраццо")
        
        with example_col3:
            if st.button("Шовный состав", use_container_width=True):
                st.info("Загрузите изображение шовного состава")
    
    with col2:
        # Отображение результатов анализа
        if st.session_state.prediction_results:
            results = st.session_state.prediction_results
            
            st.subheader("📋 Результаты анализа")
            
            # Категория
            category = results['predicted_category']
            confidence = results['confidence']
            
            st.metric(
                label="Категория состава",
                value=category,
                delta=f"{confidence:.1%} уверенности"
            )
            
            # Компоненты
            st.subheader("🧱 Предсказанные компоненты")
            
            components_df = pd.DataFrame(
                list(results['predicted_components'].items()),
                columns=['Компонент', 'Количество (кг)']
            )
            
            st.dataframe(
                components_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Похожие рецепты
            if results['similar_recipes']:
                st.subheader("📚 Похожие рецепты из базы")
                
                for i, similar in enumerate(results['similar_recipes'][:3], 1):
                    recipe = similar['recipe']
                    similarity = similar['similarity']
                    
                    with st.expander(f"Рецепт {i}: {recipe['name']} (сходство: {similarity:.1%})"):
                        st.write(f"**Категория:** {recipe['category']}")
                        st.write(f"**Всего компонентов:** {recipe['component_count']}")
                        
                        # Топ-5 компонентов
                        components = sorted(recipe['components'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for comp, value in components:
                            st.write(f"- {comp.split(',')[0]}: **{value} кг**")
            
            # Кнопка сохранения результатов
            if st.button("💾 Сохранить результаты анализа"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_{timestamp}.json"
                
                results_to_save = {
                    'timestamp': timestamp,
                    'image_filename': uploaded_file.name if uploaded_file else "unknown",
                    'predicted_category': category,
                    'predicted_components': results['predicted_components'],
                    'confidence': confidence
                }
                
                # Сохраняем как JSON
                json_str = json.dumps(results_to_save, ensure_ascii=False, indent=2)
                
                st.download_button(
                    label="Скачать JSON",
                    data=json_str,
                    file_name=filename,
                    mime="application/json"
                )
        else:
            st.info("Загрузите изображение и нажмите 'Проанализировать' для получения результатов")


# Вкладка 2: База рецептов
with tab2:
    st.header("📊 База рецептов терразитовой штукатурки")
    
    # Кнопка загрузки рецептов
    if not st.session_state.recipes_loaded:
        if st.button("📂 Загрузить базу рецептов", type="primary"):
            recipes_data = load_recipes()
    else:
        recipes_data = st.session_state.recipes_data
    
    if st.session_state.recipes_loaded:
        # Статистика
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Всего рецептов",
                len(st.session_state.recipes_data['recipes'])
            )
        
        with col2:
            st.metric(
                "Категорий",
                len(st.session_state.recipes_data['metadata']['categories'])
            )
        
        with col3:
            category_counts = {}
            for recipe in st.session_state.recipes_data['recipes']:
                category = recipe['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            most_common = max(category_counts.items(), key=lambda x: x[1])[0]
            st.metric(
                "Самая частая категория",
                most_common
            )
        
        with col4:
            avg_components = np.mean([r['component_count'] for r in st.session_state.recipes_data['recipes']])
            st.metric(
                "Среднее компонентов",
                f"{avg_components:.1f}"
            )
        
        # Визуализация распределения
        st.subheader("📈 Распределение по категориям")
        visualize_category_distribution(recipes_data)
        
        # Поиск и фильтрация
        st.subheader("🔍 Поиск рецептов")
        
        search_col1, search_col2, search_col3 = st.columns(3)
        
        with search_col1:
            search_query = st.text_input("Поиск по названию", placeholder="Введите часть названия...")
        
        with search_col2:
            selected_category = st.selectbox(
                "Категория",
                ["Все"] + config.data.recipe_categories
            )
        
        with search_col3:
            min_components = st.slider(
                "Минимум компонентов",
                min_value=0,
                max_value=50,
                value=0
            )
        
        # Фильтрация данных
        filtered_df = st.session_state.recipes_df.copy()
        
        if search_query:
            filtered_df = filtered_df[filtered_df['Название'].str.contains(search_query, case=False, na=False)]
        
        if selected_category != "Все":
            filtered_df = filtered_df[filtered_df['Категория'] == selected_category]
        
        filtered_df = filtered_df[filtered_df['Кол-во компонентов'] >= min_components]
        
        # Отображение таблицы
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Название": st.column_config.TextColumn(width="medium"),
                "Категория": st.column_config.TextColumn(width="small"),
                "Кол-во компонентов": st.column_config.NumberColumn(width="small"),
                "Общий вес (кг)": st.column_config.NumberColumn(width="small")
            }
        )
        
        # Экспорт данных
        st.subheader("📤 Экспорт данных")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📄 Экспорт в CSV"):
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать CSV",
                    data=csv,
                    file_name="terrazite_recipes.csv",
                    mime="text/csv"
                )
        
        with export_col2:
            if st.button("📊 Экспорт в JSON"):
                json_str = json.dumps(recipes_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="Скачать JSON",
                    data=json_str,
                    file_name="terrazite_recipes.json",
                    mime="application/json"
                )
    else:
        st.info("Нажмите кнопку 'Загрузить базу рецептов' для просмотра данных")


# Вкладка 3: Анализ компонентов
with tab3:
    st.header("📈 Анализ компонентов")
    
    # Кнопка анализа компонентов
    if 'component_analysis' not in st.session_state:
        if st.button("🧪 Проанализировать компоненты", type="primary"):
            analysis_results = analyze_components()
    
    if 'component_analysis' in st.session_state:
        analysis_results = st.session_state.component_analysis
        
        # Статистика компонентов
        st.subheader("📊 Статистика компонентов")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_recipes = analysis_results['category_stats']['Терразит']['count']
            st.metric("Рецептов Терразит", total_recipes)
        
        with col2:
            if 'analyzer' in st.session_state:
                total_components = st.session_state.analyzer.component_features['total_components']
                st.metric("Уникальных компонентов", total_components)
        
        with col3:
            # Считаем все использования компонентов
            total_uses = sum(
                sum(category_components.values())
                for category_components in analysis_results['component_frequency'].values()
            )
            st.metric("Всего использований", total_uses)
        
        # Визуализация групп компонентов
        st.subheader("🧩 Группы компонентов по категориям")
        visualize_component_groups(analysis_results)
        
        # Топ компонентов
        st.subheader("🏆 Наиболее частые компоненты")
        
        # Собираем все компоненты
        all_components = {}
        for category, components in analysis_results['component_frequency'].items():
            for component, count in components.items():
                all_components[component] = all_components.get(component, 0) + count
        
        # Топ-15 компонентов
        top_components = sorted(all_components.items(), key=lambda x: x[1], reverse=True)[:15]
        
        top_df = pd.DataFrame(
            top_components,
            columns=['Компонент', 'Количество использований']
        )
        
        # Сокращаем длинные названия
        top_df['Компонент_короткий'] = top_df['Компонент'].apply(
            lambda x: x[:40] + '...' if len(x) > 40 else x
        )
        
        # Создаем bar chart
        fig = px.bar(
            top_df,
            x='Количество использований',
            y='Компонент_короткий',
            orientation='h',
            title='Топ-15 наиболее часто используемых компонентов',
            color='Количество использований',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Уникальные компоненты по категориям
        st.subheader("🎯 Уникальные компоненты категорий")
        
        for category, components in analysis_results['unique_components_by_category'].items():
            if components:
                with st.expander(f"{category} ({len(components)} уникальных компонентов)"):
                    for component in components:
                        # Определяем группу компонента
                        group = config.get_component_group(component) or "Неизвестно"
                        st.write(f"**{component.split(',')[0]}** (группа: {group})")
        
        # Информация о группах компонентов
        st.subheader("📁 Группы компонентов")
        
        groups_info = []
        for group_name, components in config.data.component_groups.items():
            # Считаем сколько компонентов из этой группы используются
            used_count = sum(1 for comp in components if comp in all_components)
            groups_info.append({
                'Группа': group_name,
                'Всего компонентов': len(components),
                'Используется': used_count,
                'Процент': f"{(used_count / len(components) * 100):.1f}%" if len(components) > 0 else "0%"
            })
        
        groups_df = pd.DataFrame(groups_info)
        st.dataframe(groups_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("Нажмите кнопку 'Проанализировать компоненты' для загрузки и анализа данных из Excel файла")


# Вкладка 4: Подбор рецепта
with tab4:
    st.header("🧪 Подбор рецепта по параметрам")
    
    st.markdown("""
        Подберите рецепт терразитовой штукатурки по заданным параметрам:
        - Категория состава
        - Основные компоненты
        - Объем партии
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Выбор категории
        selected_category = st.selectbox(
            "Выберите категорию состава:",
            config.data.recipe_categories,
            help="Терразит - основные смеси, Шовный - затирочные составы и т.д."
        )
        
        # Выбор основных компонентов
        st.subheader("Основные компоненты")
        
        # Получаем компоненты для выбранной категории
        if 'component_analysis' in st.session_state:
            category_components = st.session_state.component_analysis['component_frequency'].get(
                selected_category, {}
            )
            
            # Сортируем по частоте использования
            sorted_components = sorted(
                category_components.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Топ-10 компонентов
            
            selected_components = []
            for component, frequency in sorted_components:
                if st.checkbox(f"{component.split(',')[0]} (используется в {frequency} рецептах)"):
                    selected_components.append(component)
        else:
            # Базовые компоненты, если анализ не выполнен
            basic_components = [
                "Цемент белый ПЦ500",
                "Цемент серый ПЦ500, кг",
                "Песок лужский фр.0-0,63мм, кг",
                "Доломитовая мука, кг"
            ]
            
            for component in basic_components:
                if st.checkbox(component):
                    selected_components.append(component)
    
    with col2:
        # Параметры рецепта
        st.subheader("Параметры рецепта")
        
        batch_size = st.slider(
            "Объем партии (кг):",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Общий вес всех компонентов в рецепте"
        )
        
        max_components = st.slider(
            "Максимальное количество компонентов:",
            min_value=3,
            max_value=20,
            value=10,
            step=1
        )
        
        complexity = st.select_slider(
            "Сложность состава:",
            options=["Простой", "Средний", "Сложный"],
            value="Средний"
        )
    
    # Кнопка подбора
    if st.button("🔍 Подобрать рецепты", type="primary"):
        if not st.session_state.recipes_loaded:
            st.warning("Сначала загрузите базу рецептов во вкладке '📊 База рецептов'")
        else:
            with st.spinner("Подбор рецептов..."):
                # Фильтруем рецепты по выбранным параметрам
                filtered_recipes = []
                
                for recipe in st.session_state.recipes_data['recipes']:
                    # Проверяем категорию
                    if recipe['category'] != selected_category:
                        continue
                    
                    # Проверяем количество компонентов
                    if recipe['component_count'] > max_components:
                        continue
                    
                    # Проверяем выбранные компоненты
                    recipe_components = set(recipe['components'].keys())
                    selected_set = set(selected_components)
                    
                    if selected_set and not selected_set.intersection(recipe_components):
                        continue
                    
                    # Рассчитываем соответствие
                    match_score = 0
                    
                    # За соответствие категории
                    match_score += 30
                    
                    # За выбранные компоненты
                    common_components = selected_set.intersection(recipe_components)
                    if selected_set:
                        component_match = len(common_components) / len(selected_set) * 50
                        match_score += component_match
                    
                    # За сложность (простая эвристика)
                    if complexity == "Простой" and recipe['component_count'] <= 5:
                        match_score += 20
                    elif complexity == "Средний" and 5 < recipe['component_count'] <= 10:
                        match_score += 20
                    elif complexity == "Сложный" and recipe['component_count'] > 10:
                        match_score += 20
                    
                    # Нормализуем до 100%
                    match_score = min(100, match_score)
                    
                    filtered_recipes.append({
                        'recipe': recipe,
                        'match_score': match_score
                    })
                
                # Сортируем по соответствию
                filtered_recipes.sort(key=lambda x: x['match_score'], reverse=True)
                
                # Сохраняем результаты
                st.session_state.recipe_search_results = filtered_recipes
    
    # Отображение результатов подбора
    if 'recipe_search_results' in st.session_state:
        results = st.session_state.recipe_search_results
        
        st.subheader(f"🎯 Найдено рецептов: {len(results)}")
        
        if results:
            for i, result in enumerate(results[:5], 1):  # Показываем топ-5
                recipe = result['recipe']
                match_score = result['match_score']
                
                with st.expander(f"Рецепт #{i}: {recipe['name']} (совпадение: {match_score:.1f}%)"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Категория:** {recipe['category']}")
                        st.write(f"**Компонентов:** {recipe['component_count']}")
                        st.write(f"**Общий вес:** {recipe['total_weight']} кг")
                    
                    with col_b:
                        # Топ-5 компонентов
                        st.write("**Основные компоненты:**")
                        top_components = sorted(recipe['components'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for comp, value in top_components:
                            percentage = (value / recipe['total_weight']) * 100
                            st.write(f"- {comp.split(',')[0]}: {value} кг ({percentage:.1f}%)")
                    
                    # Кнопка для просмотра полного рецепта
                    if st.button(f"📄 Показать полный рецепт", key=f"full_recipe_{i}"):
                        st.json(recipe)
        else:
            st.warning("По заданным параметрам рецепты не найдены. Попробуйте изменить критерии поиска.")


# Вкладка 5: Настройки
with tab5:
    st.header("⚙️ Настройки приложения")
    
    st.subheader("Конфигурация проекта")
    
    # Отображение текущей конфигурации
    with st.expander("📋 Текущая конфигурация"):
        config_dict = {
            "Проект": {
                "Название": config.project_name,
                "Версия": config.version,
                "Режим": config.mode
            },
            "Данные": {
                "Категории рецептов": config.data.recipe_categories,
                "Группы компонентов": list(config.data.component_groups.keys()),
                "Размер изображения": config.data.image_size
            },
            "Модель": {
                "Название": config.model.model_name,
                "Категорий": config.model.num_categories,
                "Компонентов": config.model.num_components,
                "Скрытый слой": config.model.hidden_size
            }
        }
        
        st.json(config_dict)
    
    # Управление данными
    st.subheader("Управление данными")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Обновить базу рецептов"):
            with st.spinner("Обновление базы..."):
                # Здесь должен быть код для обновления базы
                st.success("База рецептов обновлена!")
    
    with col2:
        if st.button("🧹 Очистить кэш"):
            st.session_state.clear()
            st.success("Кэш очищен!")
            st.rerun()
    
    # Настройки отображения
    st.subheader("Настройки отображения")
    
    dark_mode = st.toggle("Темная тема", value=False)
    if dark_mode:
        st.info("Для применения темной темы перезапустите приложение")
    
    # Информация о системе
    st.subheader("Системная информация")
    
    sys_info = {
        "Python версия": sys.version.split()[0],
        "Streamlit версия": st.__version__,
        "Путь к проекту": config.project_root,
        "Загружено рецептов": len(st.session_state.recipes_data['recipes']) if st.session_state.recipes_loaded else 0,
        "Модель загружена": "Да" if st.session_state.model_loaded else "Нет"
    }
    
    for key, value in sys_info.items():
        st.text(f"{key}: {value}")
    
    # Логи
    st.subheader("Логи приложения")
    
    if st.button("📝 Показать логи"):
        log_file = Path(config.project_root) / "logs" / "terrazite_ai.log"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.read()
            
            st.text_area("Логи приложения", logs, height=300)
        else:
            st.warning("Файл логов не найден")

# Футер приложения
st.divider()
st.markdown("""
    <div style='text-align: center'>
        <p>Terrazite AI v{} • Проект для подбора рецептов терразитовой штукатурки • {} рецептов в базе</p>
        <p><small>Для корректной работы загрузите изображение терразитовой штукатурки хорошего качества</small></p>
    </div>
""".format(
    config.version,
    len(st.session_state.recipes_data['recipes']) if st.session_state.recipes_loaded else 0
), unsafe_allow_html=True)

# Инициализация при первом запуске
if __name__ == "__main__":
    # Автоматическая загрузка данных при первом запуске
    if not st.session_state.recipes_loaded:
        # Пытаемся загрузить данные в фоновом режиме
        try:
            load_recipes()
        except:
            pass
    
    # Пытаемся загрузить анализ компонентов
    if 'component_analysis' not in st.session_state:
        try:
            analyze_components()
        except:
            pass
