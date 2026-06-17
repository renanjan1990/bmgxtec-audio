import streamlit as st
from faster_whisper import WhisperModel
import os
import tempfile

# Configuração da página
st.set_page_config(page_title="Transcritor de Áudio", page_icon="🎙️")

st.title("🎙️ Transcritor de Áudio Local")
st.markdown("Carregue um arquivo de áudio para transformá-lo em texto usando IA.")

# Configurações do modelo na lateral
st.sidebar.header("Configurações")
model_size = st.sidebar.selectbox(
    "Tamanho do Modelo (IA)",
    ["tiny", "base", "small", "medium"],
    index=1,
    help="Modelos maiores são mais precisos, mas demoram mais e exigem mais do PC."
)

# Upload do arquivo
uploaded_file = st.file_uploader("Escolha um arquivo de áudio...", type=["mp3", "wav", "m4a", "ogg", "flac"])

if uploaded_file is not None:
    # Mostrar player de áudio
    st.audio(uploaded_file)
    
    if st.button("Iniciar Transcrição"):
        try:
            with st.spinner("Carregando modelo e processando..."):
                # Salvar arquivo temporário para o Whisper ler
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Inicializar modelo
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                
                # Transcrever
                segments, info = model.transcribe(tmp_path, beam_size=5)
                
                st.success(f"Idioma detectado: {info.language} ({info.language_probability:.2f})")
                
                # Coletar resultados
                transcricao_texto = ""
                placeholder = st.empty() # Para mostrar o texto conforme é gerado
                
                for segment in segments:
                    linha = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    transcricao_texto += linha
                    placeholder.text_area("Resultado:", value=transcricao_texto, height=300)

                # Botão para baixar a transcrição
                st.download_button(
                    label="Baixar Transcrição (.txt)",
                    data=transcricao_texto,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcricao.txt",
                    mime="text/plain"
                )
                
                # Limpar arquivo temporário
                os.remove(tmp_path)
                
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
else:
    st.info("Aguardando upload de arquivo...")

st.markdown("---")
st.caption("Desenvolvido com Python, Streamlit e Faster-Whisper.")
