import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sistema de Predição de Obesidade",
    page_icon="🏥",
    layout="wide"
)
st.markdown("""
<style>
/* Fundo geral */
.stApp { background-color: #F4F6F9; }

/* Texto padrão */
body, p, span, label { color: #1E1E1E !important; }

/* ===== Forçar texto principal da página para preto ===== */
section.main * {
    color: #1E1E1E !important;
}

/* Corrigir listas (bullet points) */
section.main ul li {
    color: #1E1E1E !important;
}

/* Corrigir markdown renderizado */
section.main p, 
section.main span,
section.main li {
    color: #1E1E1E !important;
}

/* ===== Corrigir textos com opacidade reduzida ===== */
[data-testid="stMarkdownContainer"] {
    opacity: 1 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #1E1E1E !important;
    opacity: 1 !important;
}

[class*="secondary"] {
    color: #1E1E1E !important;
    opacity: 1 !important;
}


/* Títulos */
h1, h2, h3 { color: #0F4C8A !important; font-weight: 700; }

/* Card do Form */
div[data-testid="stForm"] {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

/* Métricas */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

/* Botão normal (fora do form) */
div.stButton > button:first-child {
    background-color: #1F5FA8;
    color: white !important;
    border-radius: 8px;
    border: none;
    height: 3em;
    font-weight: 600;
}
div.stButton > button:first-child:hover { background-color: #2E74C9; }

/* ===== Botão submit do formulário ===== */
div[data-testid="stFormSubmitButton"] button {
    background-color: #1F5FA8 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    height: 3em !important;
    font-weight: 600 !important;
    width: 100% !important;
}

/* Forçar texto interno do botão */
div[data-testid="stFormSubmitButton"] button span {
    color: white !important;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #2E74C9 !important;
    color: white !important;
}
/*Forçar o tema escuro */
div[data-testid="stNumberInput"] input {
    background-color: white !important;
    color: #1E1E1E !important;
}
div[data-testid="stNumberInput"] button {
    background-color: #E9EEF4 !important;
    color: #1E1E1E !important;
    border: none !important;
}
/* ===== Corrigir seta da Selectbox ===== */

/* Ícone da seta */
div[data-baseweb="select"] svg {
    fill: #1E1E1E !important;
}

/* Área do select */
div[data-baseweb="select"] {
    border: 1px solid #D0D7E2 !important;
    border-radius: 6px !important;
}

/* Hover elegante */
div[data-baseweb="select"]:hover {
    border: 1px solid #1F5FA8 !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #1E1E1E !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #0F4C8A; }
section[data-testid="stSidebar"] * { color: white !important; }

</style>
""", unsafe_allow_html=True)


TARGET_COL = "Obesity"
MODEL_PATH = "modelo_obesidade_pipeline.joblib"
DATA_FE_PATH = "obesity_fe.csv"   # preferível
DATA_RAW_PATH = "obesity.csv"     # fallback

# =========================
# LOADERS
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    # tenta FE primeiro
    try:
        df = pd.read_csv(DATA_FE_PATH)
        return df, True
    except Exception:
        df = pd.read_csv(DATA_RAW_PATH)
        # cria BMI se não existir
        if "BMI" not in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        return df, False

def obesity_group(label: str) -> str:
    if label.startswith("Obesity_"):
        return "Obesidade"
    if label.startswith("Overweight_"):
        return "Sobrepeso"
    if label == "Normal_Weight":
        return "Peso normal"
    if label == "Insufficient_Weight":
        return "Abaixo do peso"
    return "Outro"

# =========================
# UI - HEADER
# =========================
st.title("🏥 Sistema de Apoio à Decisão: Predição de Obesidade")
st.caption("Modelo de Machine Learning (RandomForest + BMI) para auxiliar triagem e priorização clínica. Não substitui diagnóstico médico.")

# sidebar navigation
page = st.sidebar.radio(
    "Navegação",
    ["📌 Predição (Triagem)", "📊 Dashboard (Insights)", "ℹ️ Sobre o Modelo"],
    index=0
)

# load resources
model = load_model()
df, loaded_fe = load_data()

# =========================
# PAGE 1: PREDIÇÃO
# =========================
if page == "📌 Predição (Triagem)":
    st.subheader("📌 Triagem individual")

    with st.form("triagem_form"):

        st.markdown("### 👤 Informações Básicas")
    
        c1, c2 = st.columns(2)
    
        with c1:
            Gender = st.selectbox(
                "Gênero",
                sorted(df["Gender"].dropna().unique().tolist())
            )
    
            Age = st.number_input(
                "Idade",
                min_value=1, max_value=120, value=30
            )
    
            Height = st.number_input(
                "Altura (em metros)",
                min_value=1.0, max_value=2.5, value=1.70, step=0.01
            )
    
            Weight = st.number_input(
                "Peso (em kg)",
                min_value=20.0, max_value=300.0, value=75.0, step=0.5
            )
    
        with c2:
            family_history = st.selectbox(
                "Algum membro da família sofreu ou sofre de excesso de peso?",
                sorted(df["family_history"].dropna().unique().tolist())
            )
    
            SMOKE = st.selectbox(
                "Você fuma?",
                sorted(df["SMOKE"].dropna().unique().tolist())
            )
    
            MTRANS = st.selectbox(
                "Qual meio de transporte você costuma usar?",
                sorted(df["MTRANS"].dropna().unique().tolist())
            )
    
        st.markdown("### 🍎 Hábitos Alimentares")
    
        c3, c4 = st.columns(2)
    
        with c3:
            FAVC = st.selectbox(
                "Você come alimentos altamente calóricos com frequência?",
                sorted(df["FAVC"].dropna().unique().tolist())
            )
    
            FCVC = st.number_input(
                "Você costuma comer vegetais nas suas refeições? (0 a 3)",
                min_value=0.0, max_value=3.0, value=2.0, step=0.1
            )
    
            NCP = st.number_input(
                "Quantas refeições principais você faz diariamente?",
                min_value=0.0, max_value=5.0, value=3.0, step=0.1
            )
    
        with c4:
            CAEC = st.selectbox(
                "Você come alguma coisa entre as refeições?",
                sorted(df["CAEC"].dropna().unique().tolist())
            )
    
            SCC = st.selectbox(
                "Você monitora as calorias que ingere diariamente?",
                sorted(df["SCC"].dropna().unique().tolist())
            )
    
            CH2O = st.number_input(
                "Quanta água você bebe diariamente? (0 a 5)",
                min_value=0.0, max_value=5.0, value=2.0, step=0.1
            )
    
        st.markdown("### 🏃 Estilo de Vida")
    
        c5, c6 = st.columns(2)
    
        with c5:
            FAF = st.number_input(
                "Com que frequência você pratica atividade física? (0 a 3)",
                min_value=0.0, max_value=3.0, value=1.0, step=0.1
            )
    
        with c6:
            TUE = st.number_input(
                "Quanto tempo você usa dispositivos tecnológicos diariamente? (0 a 3)",
                min_value=0.0, max_value=3.0, value=1.0, step=0.1
            )
    
            CALC = st.selectbox(
                "Com que frequência você bebe álcool?",
                sorted(df["CALC"].dropna().unique().tolist())
            )
    
        submitted = st.form_submit_button("🔎 Analisar risco de obesidade")
    
        if submitted:
            bmi = Weight / (Height ** 2)
    
            # montar input exatamente com colunas do treino + BMI
            row = {
                "Gender": Gender,
                "Age": Age,
                "Height": Height,
                "Weight": Weight,
                "family_history": family_history,
                "FAVC": FAVC,
                "FCVC": FCVC,
                "NCP": NCP,
                "CAEC": CAEC,
                "SMOKE": SMOKE,
                "CH2O": CH2O,
                "SCC": SCC,
                "FAF": FAF,
                "TUE": TUE,
                "CALC": CALC,
                "MTRANS": MTRANS,
                "BMI": bmi,
            }
            X_input = pd.DataFrame([row])
    
            pred = model.predict(X_input)[0]
            group = obesity_group(pred)
    
            # Probabilidades (se o modelo suportar)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                classes = model.classes_
                proba_df = pd.DataFrame({"classe": classes, "probabilidade": proba}).sort_values("probabilidade", ascending=False)
    
            k1, k2, k3 = st.columns(3)
            k1.metric("Resultado previsto", pred)
            k2.metric("Categoria clínica", group)
            k3.metric("BMI (IMC)", f"{bmi:.2f}")
    
            if group == "Obesidade":
                st.error("⚠️ Perfil de maior risco. Recomenda-se avaliação clínica e plano de intervenção.")
            elif group == "Sobrepeso":
                st.warning("⚠️ Perfil de atenção. Recomenda-se prevenção e monitoramento.")
            elif group == "Abaixo do peso":
                st.info("ℹ️ Perfil abaixo do peso. Avaliar risco nutricional.")
            else:
                st.success("✅ Perfil dentro do esperado para peso normal.")
    
            if proba is not None:
                st.markdown("### Probabilidade por classe")
                fig = px.bar(proba_df.head(7), x="classe", y="probabilidade")
                st.plotly_chart(fig, use_container_width=True)

# =========================
# PAGE 2: DASHBOARD
# =========================
elif page == "📊 Dashboard (Insights)":
    st.subheader("📊 Painel Analítico — insights para a equipe médica")

    # filtros
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        genders = ["Todos"] + sorted(df["Gender"].dropna().unique().tolist())
        gender_sel = st.selectbox("Filtrar por Gender", genders, index=0)
    with colf2:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Faixa etária", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    with colf3:
        groups = ["Todos", "Obesidade", "Sobrepeso", "Peso normal", "Abaixo do peso"]
        group_sel = st.selectbox("Filtrar por grupo clínico", groups, index=0)

    dff = df.copy()
    dff["grupo_clinico"] = dff[TARGET_COL].apply(obesity_group)

    if gender_sel != "Todos":
        dff = dff[dff["Gender"] == gender_sel]
    dff = dff[(dff["Age"] >= age_range[0]) & (dff["Age"] <= age_range[1])]
    if group_sel != "Todos":
        dff = dff[dff["grupo_clinico"] == group_sel]

    # KPIs
    total = len(dff)
    pct_obes = (dff["grupo_clinico"].eq("Obesidade").mean() * 100) if total else 0
    pct_sobre = (dff["grupo_clinico"].eq("Sobrepeso").mean() * 100) if total else 0
    bmi_mean = dff["BMI"].mean() if total else 0
    age_mean = dff["Age"].mean() if total else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total (filtro)", f"{total}")
    k2.metric("% Obesidade", f"{pct_obes:.1f}%")
    k3.metric("% Sobrepeso", f"{pct_sobre:.1f}%")
    k4.metric("BMI médio", f"{bmi_mean:.2f}")
    k5.metric("Idade média", f"{age_mean:.1f}")

    st.divider()

    # Gráficos
    c1, c2 = st.columns(2)

    with c1:
        dist = dff[TARGET_COL].value_counts().reset_index()
        dist.columns = ["classe", "contagem"]
        fig = px.bar(dist, x="classe", y="contagem", title="Distribuição dos níveis de obesidade")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_gender = (
            dff.groupby(["Gender", TARGET_COL])
              .size()
              .reset_index(name="contagem")
        )
        fig = px.bar(by_gender, x="Gender", y="contagem", color=TARGET_COL,
                     title="Distribuição por gênero x nível")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.box(dff, x=TARGET_COL, y="Age", title="Idade por nível de obesidade")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.box(dff, x=TARGET_COL, y="BMI", title="BMI (IMC) por nível de obesidade")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Hábitos e estilo de vida (associação com obesidade)")

    c5, c6, c7 = st.columns(3)

    with c5:
        favc = pd.crosstab(dff[TARGET_COL], dff["FAVC"], normalize="index").reset_index()
        fig = px.bar(favc, x=TARGET_COL, y=favc.columns[1:], barmode="stack",
                     title="FAVC (alimentos calóricos) por nível")
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        faf = dff.groupby(TARGET_COL)["FAF"].mean().reset_index()
        fig = px.bar(faf, x=TARGET_COL, y="FAF", title="FAF (atividade física média) por nível")
        st.plotly_chart(fig, use_container_width=True)

    with c7:
        tue = dff.groupby(TARGET_COL)["TUE"].mean().reset_index()
        fig = px.bar(tue, x=TARGET_COL, y="TUE", title="TUE (tempo tecnologia médio) por nível")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Importâncias do modelo (RandomForest)
    st.markdown("### Top variáveis do modelo (explicabilidade)")
    try:
        rf_model = model.named_steps["model"]
        pre = model.named_steps["preprocess"]
        feature_names = pre.get_feature_names_out()

        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(15).reset_index()
        top.columns = ["feature", "importance"]

        fig = px.bar(top[::-1], x="importance", y="feature", orientation="h",
                     title="Top 15 features por importância (RandomForest)")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Interpretação: maior importância indica maior contribuição do atributo para a decisão do modelo.")
    except Exception as e:
        st.info("Não foi possível gerar importâncias automaticamente (depende do pipeline).")

# =========================
# PAGE 3: SOBRE
# =========================
else:
    st.subheader("ℹ️ Sobre o modelo e a estratégia")

    st.markdown("""
**Objetivo:** prever o *nível de obesidade* (7 classes) para apoiar a triagem clínica.

**Modelo:** RandomForest com pipeline completo (tratamento de numéricas e categóricas) + feature engineering com **BMI (IMC)**.
""")

    st.markdown("""
### Desempenho (seu notebook)

<ul style="color:#1E1E1E;">
  <li><b>Accuracy teste:</b> 0.976</li>
  <li><b>CV accuracy:</b> 0.985 (+/- 0.008)</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("""
**Boas práticas:**
- O sistema é suporte à decisão e não substitui avaliação médica.
- Recomenda-se monitorar drift e revalidar o modelo periodicamente.
""")
st.markdown("""
### Estratégia de Modelagem

O modelo foi desenvolvido utilizando uma pipeline completa de Machine Learning, incluindo:

- Tratamento de valores ausentes
- Padronização de variáveis numéricas
- Codificação de variáveis categóricas
- Engenharia de atributos (BMI)
- Classificador RandomForest

### Justificativa da Escolha do Modelo

O RandomForest foi escolhido por sua robustez para dados tabulares, capacidade de capturar relações não lineares e bom desempenho em problemas multiclasse.

### Validação

- Validação cruzada (5-fold): 98,5% de acurácia média
- Teste holdout: 97,6% de acurácia

A baixa diferença entre validação e teste indica boa capacidade de generalização.

### Considerações

O sistema atua como apoio à decisão clínica e não substitui avaliação médica especializada.
""")