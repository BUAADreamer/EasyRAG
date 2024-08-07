# -*- coding: UTF-8 -*-
import requests
import streamlit as st

root = "http://localhost:8000/v1"


def post_request(data, endpoint):
    url = f"{root}/{endpoint}"
    response = requests.post(url, json=data)
    res = response.json()
    return res


st.markdown(
    "## 📖EasyRAG智能助手\n"
    "1. 输入问题。\n"
    "2. 等待智能助手返回答案💬\n"
)
with st.form(key="qa_form"):
    query = st.text_area("输入问题:")
    document = st.selectbox(
        "选择文档来源(可选):", options=list(['无', 'director', 'emsplus', 'rcp', 'umac'])
    )
    submit = st.form_submit_button("开始回答")

if submit:
    with st.spinner('生成中...'):
        data = {'query': query}
        if document is not None and document != '无':
            data['document'] = document
        try:
            res = post_request(data, endpoint="rag")
        except:
            res = {
                "answer": "你好!",
                "contexts": ['你好!', '你好!']
            }

    st.markdown(f'''## 答案💬''')
    st.markdown(f"{res['answer']}")

    contexts = res['contexts']
    st.markdown(f'''## 文档列表📄''')
    for i, context in enumerate(contexts):
        with st.expander(f"文档{i}"):
            st.markdown(f"{context}")
