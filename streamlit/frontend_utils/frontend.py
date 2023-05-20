import streamlit as st
import requests
from requests.structures import CaseInsensitiveDict
import pyperclip

API_URL = "http://localhost:8000/summarize/"


def summarize(
        text: str, min_length: int, max_length: int, length_penalty: float, 
        no_repeat_ngram_size: int, repetition_penalty: float, top_k: int, top_p: float,
        num_beams: int, temperature: float, model_name: str
) -> str:
    input_json = {
        "text": text, 
        "min_length": min_length, 
        "max_length": max_length, 
        "length_penalty": length_penalty, 
        "no_repeat_ngram_size": no_repeat_ngram_size, 
        "repetition_penalty": repetition_penalty, 
        "top_k": top_k, 
        "top_p": top_p, 
        "num_beams": num_beams, 
        "temperature": temperature, 
        "model_name": model_name
        }
    
    response = requests.post(API_URL, json=input_json)

    if response.status_code != 200:
        raise ValueError(f"Ошибка генерации. Код статуса: {response.status_code}")
    
    return response.json()["summary"]

st.set_page_config(
    page_title="Нейросетевое аннотирование текста",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.title("Нейросетевое аннотирование текста",)


header1 = st.header("Введите текст статьи:")
text = st.text_area("Введите текст статьи:", label_visibility="collapsed")

model_name = st.selectbox("Выберите модель генерации:", options=["ruT5-base", "FRED-T5-Large", "mBART"])

sum_button = st.button("Создать аннотацию")

with st.expander("Дополнительные параметры генерации"):
    st.markdown("Рекомендация к длине аннотации. При :green[length_penalty>0] будут поощряться более длинные тексты, при :green[length_penalty<0] - более короткие.")
    length_penalty = st.number_input("length_penalty", min_value=-10.0, max_value=20.0, value=1.0, step=0.1)
    st.divider()
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Контроль повторений. Чем выше значение, тем реже будут встречаться повторения. При :green[repetition_penalty=1] штрафа за повторения нет.")
        repetition_penalty = st.number_input("repetition_penalty:", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        st.divider()
    
        st.markdown("Строгое ограничений повторений. При :green[no_repeat_ngram_size>0] последовательности символов соответсвующей длины не будут иметь возможности повторения.")
        no_repeat_ngram_size = st.number_input("no_repeat_ngram_size:", min_value=0, max_value=10, value=3, step=1)
        st.divider()

        st.markdown("Ограничения генерируемого текста по числу токенов. В среднем слово кодируется 1-3 токенами.")
        min_length = st.slider("min_length:", min_value=10, max_value=100, value=40, step=1)
        max_length = st.slider("max_length:", min_value=min_length, max_value=500, value=150, step=1)
        st.divider()

    with col2:
        st.markdown("Ограничение числа наиболее вероятных кандидатов при генерации следующего токена. :green[top_k] указывает на число кандидатов, а :green[top_p] отсекает долю наименее вероятных вариантов по статистическую вероятность.")
        top_k = st.number_input("top_k:", min_value=0, max_value=100, value=50, step=1)
        top_p = st.number_input("top_p:", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        st.divider()

        st.markdown('Количество "сценариев", рассматриваемых моделью при аннотировании, для определения лучшей последовательности слов. Увеличение :green[num_beams] может улучшить результат, но замедлит процесс. При :green[num_beams=1] генерация будет жадной.')
        num_beams = st.number_input("num_beams:", min_value=1, max_value=10, value=4, step=1)
        st.divider()
    
        st.markdown("Контроль разнообразия генерации. Чем ниже :green[temperature], тем детерменированнее и надежнее будет результат. Чем выше, тем нестабильнее, но многообразнее будет генерация.")
        temperature = st.number_input("temperature:", min_value=0.1, max_value=10.0, value=0.7, step=0.1)



if 'summary' not in st.session_state:
    st.session_state['summary'] = ''

if 'recent_summaries' not in st.session_state:
    st.session_state['recent_summaries'] = []

if 'recent_params' not in st.session_state:
    st.session_state['recent_params'] = []

if sum_button:
    summary = summarize(text, min_length, max_length, length_penalty, 
                        no_repeat_ngram_size, repetition_penalty, top_k, top_p,
                        num_beams, temperature, model_name)
    # summary = 'example text.'
    st.session_state['summary'] = summary
    tmp_params = {
        "min_length": min_length, 
        "max_length": max_length, 
        "length_penalty": length_penalty, 
        "no_repeat_ngram_size": no_repeat_ngram_size, 
        "repetition_penalty": repetition_penalty, 
        "top_k": top_k, 
        "top_p": top_p, 
        "num_beams": num_beams, 
        "temperature": temperature, 
        "model_name": model_name
    }

    st.session_state['recent_summaries'].insert(0, summary)
    st.session_state['recent_params'].insert(0, tmp_params)

    if len(st.session_state['recent_summaries']) > 10:
        st.session_state['recent_summaries'] = st.session_state['recent_summaries'][:10]
        st.session_state['recent_params'] = st.session_state['recent_params'][:10]



header2 = st.header("Аннотация:")
if st.session_state['summary'] != '':
    st.divider()
    st.write(st.session_state['summary'])
    st.divider()

copy_sum_button = st.button("Скопировать результат")
if copy_sum_button:
    if st.session_state['summary'] == '':
        st.error('Сначала сгенерируйте аннотацию.', icon="❌")
    else:
        pyperclip.copy(st.session_state['summary'])
        st.success('Скопировано!', icon="✅")
    
    
with st.expander("Сохранить"):
    if st.session_state['summary'] != '':
        sum_file_name = st.text_input('Укажите название файла без расширения.', max_chars=50, placeholder='annotation')
        sum_download_button = st.download_button('Скачать аннотацию в формате .txt', data=st.session_state['summary'], file_name=sum_file_name+'.txt')
    else:
        st.error('Сначала сгенерируйте аннотацию.', icon="❌")

# if st.session_state['recent_summaries']:
header3 = st.header("Последние генерации:")
for recent_sum, recent_params in zip(st.session_state['recent_summaries'], st.session_state['recent_params']):
    st.write(recent_sum)
    with st.expander("Параметры"):
        for param, value in recent_params.items():
            st.write(param + ': ' + str(value))
    st.divider()

