# SmartVideoInspector MVP
# Streamlit приложение для анализа видео с поддержкой пользовательских критериев, базового CV и AI-подсказок

import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartVideoInspector", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size:36px;
        font-weight:bold;
        color:#1F4E79;
        text-align:center;
        margin-bottom:20px;
    }
    </style>
    <div class=\"main-title\">SmartVideoInspector: AI-анализ видео процессов и ремонта</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

uploaded_video = st.file_uploader("\U0001F4C2 Загрузите видео для анализа:", type=["mp4", "mov"])

st.sidebar.title("\U0001F4DD Настройки анализа")
track_movement = st.sidebar.checkbox("\u2705 Обнаружение задержек в движении", value=True)
movement_threshold = st.sidebar.slider("Порог чувствительности", 10, 100, 30)
full_analysis = st.sidebar.checkbox("\U0001F52C Полный анализ всего видео", value=False)
user_prompt = st.sidebar.text_input("\U0001F4AC Что вас интересует на видео?", "Посчитай сколько раз появился объект в зоне")

st.sidebar.markdown("**Выберите шаблон анализа:**")
template = st.sidebar.selectbox("Шаблон", ["По умолчанию", "Контроль остановок", "Анализ сборки", "Обнаружение вращения"])

st.sidebar.markdown("**Выделите зону интереса на первом кадре:**")
use_zone = st.sidebar.checkbox("Учитывать зону интереса")
zone_selected = False

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.video(uploaded_video)
    st.info(f"\U0001F3AC Видео содержит {frame_count} кадров при {fps} FPS")

    ret, preview_frame = cap.read()
    if ret:
        st.subheader("Выбор зоны интереса (только для первого кадра)")
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=Image.fromarray(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=height,
            width=width,
            drawing_mode="rect",
            key="canvas_zone"
        )

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            x1 = int(obj["left"])
            y1 = int(obj["top"])
            x2 = x1 + int(obj["width"])
            y2 = y1 + int(obj["height"])
            zone_selected = True

    stframe = st.empty()

    prev_gray = None
    log = []
    event_descriptions = []
    total_movement = []
    movement_timeline = []

    with st.spinner("\U0001F52C Анализируем видео..."):
        for i in range(0, frame_count, int(fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            if use_zone and zone_selected:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                roi = diff[y1:y2, x1:x2] if use_zone and zone_selected else diff
                non_zero_count = np.count_nonzero(roi > 25)
                total_movement.append(non_zero_count)
                movement_timeline.append((i, non_zero_count))

                if track_movement and non_zero_count < movement_threshold * 100:
                    ts = str(datetime.now().time())
                    description = f"На кадре {i} обнаружена возможная задержка в зоне."
                    log.append((ts, i, "Задержка движения"))
                    event_descriptions.append(description)
                    cv2.putText(frame, "Задержка!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            prev_gray = gray

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("\U0001F4C4 Обнаруженные события")
    if log:
        for entry in log:
            st.write(f"[{entry[0]}] Кадр {entry[1]} — {entry[2]}")
    else:
        st.success("\U0001F389 Аномалии не обнаружены")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("\U0001F4A1 AI-рекомендации")
    if event_descriptions:
        for desc in event_descriptions:
            st.write(f"➡️ {desc}\n**Совет:** Используйте шаблон '{template}' для дальнейшего анализа. Проверьте узкие места и соответствие нормам.")
    elif full_analysis:
        avg_movement = np.mean(total_movement) if total_movement else 0
        st.info("AI-анализ общего движения:")
        if avg_movement < 1000:
            st.write("Общая активность на видео низкая. Возможно, процесс требует автоматизации или ускорения этапов.")
        elif avg_movement > 5000:
            st.write("Обнаружена высокая активность. Проверьте, нет ли хаотичных операций или перегрузки исполнителей.")
        else:
            st.write("Процесс выглядит сбалансированным. Нет явных отклонений в активности.")
        st.markdown("---")
        st.subheader("\U0001F4CA График активности")
        if movement_timeline:
            x_vals, y_vals = zip(*movement_timeline)
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label="Активность")
            ax.set_xlabel("Кадры")
            ax.set_ylabel("Уровень движения")
            ax.set_title("Анализ активности по времени")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("События не зафиксированы — рекомендации не сформированы.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("\U0001F4BB Посети [gptonline.ai](https://gptonline.ai/) для новых модулей и дополнений!")
