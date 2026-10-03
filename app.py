import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-top: 20px;
        text-align: center;
    }

    .prediction-word {
        font-size: 35px;
        font-weight: bold;
    }

    .generated-text {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        font-size: 22px;
        line-height: 1.6;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# =========================================================
# CHECK FILES
# =========================================================

missing_files = []

if not MODEL_PATH.exists():
    missing_files.append("lstm_model.h5")

if not TOKENIZER_PATH.exists():
    missing_files.append("tokenizer.pkl")

if not MAX_LEN_PATH.exists():
    missing_files.append("max_len.pkl")


if missing_files:

    st.error(
        "The following files are missing:\n\n"
        + "\n".join(
            f"- {file}" for file in missing_files
        )
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_files():

    # Load LSTM model
    model = load_model(
        MODEL_PATH,
        compile=False
    )

    # Load tokenizer
    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    # Load maximum sequence length
    with open(
        MAX_LEN_PATH,
        "rb"
    ) as file:

        max_len = pickle.load(file)

    return model, tokenizer, max_len


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    model, tokenizer, max_len = (
        load_model_files()
    )

except Exception as e:

    st.error(
        "Error while loading the model files."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🧠 LSTM Next Word Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the next word using a trained LSTM language model'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of suggestions",
    min_value=1,
    max_value=10,
    value=5
)

num_words = st.sidebar.slider(
    "Words to generate",
    min_value=1,
    max_value=10,
    value=5
)


st.sidebar.markdown("---")

st.sidebar.write(
    f"**Maximum sequence length:** {max_len}"
)

st.sidebar.write(
    f"**Vocabulary size:** {len(tokenizer.word_index)}"
)


# =========================================================
# NEXT WORD FUNCTION
# =========================================================

def predict_next_words(
    text,
    top_k=5
):

    # Remove extra spaces
    text = " ".join(
        text.strip().split()
    )

    if not text:

        return []

    # Convert text to token numbers
    token_list = tokenizer.texts_to_sequences(
        [text]
    )[0]

    if not token_list:

        return []

    # Keep only the latest max_len tokens
    token_list = token_list[-max_len:]

    # Pad sequence
    padded_sequence = pad_sequences(
        [token_list],
        maxlen=max_len,
        padding="pre",
        truncating="pre"
    )

    # Predict
    predictions = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Get top indices
    top_indices = np.argsort(
        predictions
    )[-top_k:][::-1]

    results = []

    for index in top_indices:

        # Ignore unknown/zero index
        if index == 0:
            continue

        # Find word from tokenizer
        word = None

        for token, token_index in (
            tokenizer.word_index.items()
        ):

            if token_index == index:

                word = token
                break

        if word is None:
            continue

        probability = float(
            predictions[index]
        )

        results.append(
            {
                "word": word,
                "probability": probability
            }
        )

    return results


# =========================================================
# GENERATE MULTIPLE WORDS
# =========================================================

def generate_text(
    seed_text,
    number_of_words
):

    generated_text = (
        seed_text.strip()
    )

    for _ in range(number_of_words):

        predictions = predict_next_words(
            generated_text,
            top_k=1
        )

        if not predictions:
            break

        next_word = predictions[0]["word"]

        generated_text += (
            " " + next_word
        )

    return generated_text


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("---")

st.header("✍️ Enter Your Text")

user_text = st.text_area(
    "Start typing your sentence:",
    placeholder=(
        "Example: I want to learn"
    ),
    height=140
)


# =========================================================
# BUTTON
# =========================================================

predict_button = st.button(
    "🔮 Predict Next Word",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not user_text.strip():

        st.warning(
            "Please enter some text first."
        )

        st.stop()


    predictions = predict_next_words(
        user_text,
        top_k=top_k
    )


    if not predictions:

        st.error(
            "The model could not generate a prediction "
            "for this text."
        )

        st.info(
            "Try using words that are present in "
            "the training vocabulary."
        )

        st.stop()


    # =====================================================
    # BEST PREDICTION
    # =====================================================

    best_prediction = predictions[0]

    st.markdown(
        """
        <div class="result-box">
        <div>Predicted Next Word</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-word">
        {best_prediction["word"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Model Confidence",
        f"{best_prediction['probability']:.2%}"
    )


    # =====================================================
    # TOP SUGGESTIONS
    # =====================================================

    st.markdown("---")

    st.header("💡 Top Word Suggestions")


    for i, result in enumerate(
        predictions,
        start=1
    ):

        col1, col2 = st.columns(
            [4, 1]
        )

        col1.write(
            f"**{i}. {result['word']}**"
        )

        col2.write(
            f"{result['probability']:.2%}"
        )

        st.progress(
            min(
                result["probability"],
                1.0
            )
        )


    # =====================================================
    # ONE WORD COMPLETION
    # =====================================================

    st.markdown("---")

    st.header("📝 Sentence Completion")

    completed_sentence = (
        user_text.strip()
        + " "
        + best_prediction["word"]
    )

    st.markdown(
        f"""
        <div class="generated-text">
        {completed_sentence}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # MULTI-WORD GENERATION
    # =====================================================

    st.markdown("---")

    st.header("🚀 Generate Multiple Words")

    generated_sentence = generate_text(
        user_text,
        num_words
    )

    st.markdown(
        f"""
        <div class="generated-text">
        {generated_sentence}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("---")

st.header("📊 Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Sequence Length",
    str(max_len)
)

col2.metric(
    "Vocabulary",
    f"{len(tokenizer.word_index):,}"
)

col3.metric(
    "Model Type",
    "LSTM"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "LSTM Next Word Prediction | "
    "TensorFlow + Keras + Streamlit"
)