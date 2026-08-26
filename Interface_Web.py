import streamlit as st
import datetime
import os
import math
import hashlib
import io
import numpy as np
from PIL import Image, ImageDraw

# -----------------------------------------------------------------------------
# Componente de clique na imagem
# -----------------------------------------------------------------------------
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None

# -----------------------------------------------------------------------------
# Importação dos módulos locais
# -----------------------------------------------------------------------------
try:
    from Estimação_altura import analisar_arvores_sombras, calcular_gsd_sasplanet
    from Validação import analisar_arvores_hibrido
    from pysolar.solar import get_altitude, get_azimuth
except ImportError as e:
    st.error(
        f"Erro ao importar módulos locais: {e}. "
        "Certifique-se de que 'Estimação_altura.py' e 'Validação.py' "
        "estão no mesmo diretório."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Funções auxiliares APENAS da interface
# -----------------------------------------------------------------------------
def preparar_imagem_cliques(img_original, pontos_reais, largura_max=1000):
    """
    Cria uma cópia redimensionada apenas para exibição na interface.
    Os pontos são armazenados na resolução ORIGINAL e convertidos para
    a resolução exibida somente para desenhar os marcadores.
    """
    largura_original, altura_original = img_original.size

    if largura_original > largura_max:
        escala_display = largura_max / largura_original
        largura_display = largura_max
        altura_display = max(1, round(altura_original * escala_display))
        img_display = img_original.resize(
            (largura_display, altura_display), Image.Resampling.LANCZOS
        )
    else:
        largura_display = largura_original
        altura_display = altura_original
        img_display = img_original.copy()

    escala_x = largura_original / largura_display
    escala_y = altura_original / altura_display

    desenho = ImageDraw.Draw(img_display)
    raio = max(5, round(largura_display * 0.006))

    pontos_display = []
    for x_real, y_real in pontos_reais:
        x_disp = int(round(x_real / escala_x))
        y_disp = int(round(y_real / escala_y))
        pontos_display.append((x_disp, y_disp))

    for i, (x_disp, y_disp) in enumerate(pontos_display):
        desenho.ellipse(
            (x_disp - raio, y_disp - raio, x_disp + raio, y_disp + raio),
            fill="red",
            outline="white",
            width=2,
        )
        desenho.text((x_disp + raio + 3, y_disp - raio), str(i + 1), fill="red")

    if len(pontos_display) == 2:
        desenho.line([pontos_display[0], pontos_display[1]], fill="red", width=3)

    return img_display, escala_x, escala_y, largura_display


def calcular_dados_referencia(pontos):
    """Calcula distância em pixels e azimute Base -> Ponta na imagem original."""
    if len(pontos) != 2:
        return 0.0, 0.0

    (x1, y1), (x2, y2) = pontos
    distancia_px = float(
        np.linalg.norm(np.array([x1, y1], dtype=float) - np.array([x2, y2], dtype=float))
    )

    dx = x2 - x1
    dy = y2 - y1
    azimute = (math.degrees(math.atan2(dx, -dy))) % 360

    return distancia_px, azimute


# -----------------------------------------------------------------------------
# Configuração Visual da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Estimativa de Altura de Árvores",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌳 Sistema de Estimativa de Altura de Árvores")
st.markdown(
    "Ferramenta para análise e predição de altura de árvores utilizando "
    "imagens de satélite/drone e inferência YOLO."
)

st.divider()

# -----------------------------------------------------------------------------
# Barra Lateral
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações do Modelo YOLO")
    usar_modelo_padrao = st.checkbox(
        "Usar modelo padrão ('Melhor_modelo_05-12-23.pt')", value=True
    )

    modelo_file = None
    if not usar_modelo_padrao:
        modelo_file = st.file_uploader("Upload do Modelo YOLO (.pt)", type=["pt"])

# -----------------------------------------------------------------------------
# 1. Método de Análise
# -----------------------------------------------------------------------------
st.subheader("1. Método de Análise")
modo_metodo = st.radio(
    "Método de cálculo:",
    [
        "Estimativa Sol/GSD (Geográfica e Solar)",
        "Verificação por Referência (Marcar Sombra)",
    ],
    horizontal=False,
)

st.divider()

# -----------------------------------------------------------------------------
# 2. Parâmetros conforme o método escolhido
# -----------------------------------------------------------------------------
h_ref = 4.0

if modo_metodo == "Estimativa Sol/GSD (Geográfica e Solar)":
    st.subheader("2. Escala, Localização e Horário")

    st.info(
            "ℹ️ **Modo Sol/GSD:** as posições das sombras e alturas são "
            "calculadas via Pysolar + GSD."
    )

    modo_escala = st.radio(
        "Escala espacial:",
        ["Zoom (Satélite)", "GSD (m/px)"],
        horizontal=True,
    )

    col_lat, col_long, col_escala = st.columns(3)

    with col_lat:
        lat = st.number_input(
            "Latitude (Graus Decimais)*", value=-15.7599053, format="%.7f"
        )

    with col_long:
        long = st.number_input(
            "Longitude (Graus Decimais)*", value=-47.8713185, format="%.7f"
        )

    with col_escala:
        if modo_escala == "Zoom (Satélite)":
            zoom = st.number_input(
                "Nível de Zoom (Satélite)*",
                value=22,
                step=1,
                min_value=17,
                max_value=23,
            )
            gsd_direto = None
        else:
            gsd_direto = st.number_input(
                "Valor do GSD (Metros/Pixel)*",
                value=0.05,
                format="%.5f",
                step=0.001,
            )
            zoom = 22

    col_data, col_hora, col_fuso = st.columns(3)
    with col_data:
        data_captura = st.date_input(
            "Data da Captura*", datetime.date(2026, 4, 27)
        )
    with col_hora:
        hora_captura = st.time_input(
            "Horário da Captura (Local)*", datetime.time(8, 30, 0)
        )
    with col_fuso:
        fuso = st.number_input(
            "Fuso Horário Em Relação ao UTC*", value=-3, step=1
        )

else:
    st.subheader("2. Referência de Validação")
    st.info(
        "ℹ️ **Modo Referência:** informe a altura real do poste/objeto e, "
        "na imagem, marque primeiro a **base** e depois a **ponta da sombra**. "
        "Os demais dados serão obtidos automaticamente."
    )

    h_ref = st.number_input(
        "Altura Real da referência (Metros)*",
        value=4.0,
        min_value=0.1,
        step=0.5,
    )

st.divider()

# -----------------------------------------------------------------------------
# 4. Upload e demarcação
# -----------------------------------------------------------------------------
st.subheader("4. Seleção da Imagem e Demarcação")

imagem_file = st.file_uploader(
    "Envie a Imagem para Análise (PNG, JPG, JPEG)*",
    type=["png", "jpg", "jpeg"],
)

pts_poste = None
distancia_sombra_px = 0.0
azimute_calculado = 0.0

if imagem_file is not None:
    # Lê os bytes uma única vez. Isso também permite identificar quando
    # o usuário troca a imagem e limpar apenas os pontos da imagem anterior.
    imagem_bytes = imagem_file.getvalue()
    imagem_id = hashlib.md5(imagem_bytes).hexdigest()
    img_pil = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")

    if modo_metodo == "Verificação por Referência (Marcar Sombra)":
        if streamlit_image_coordinates is None:
            st.error(
                "Para marcar pontos diretamente na imagem, instale o componente "
                "`streamlit-image-coordinates`."
            )
            st.code("pip install streamlit-image-coordinates")
            st.stop()

        # Estado persistente entre os reruns do Streamlit.
        if st.session_state.get("validacao_imagem_id") != imagem_id:
            st.session_state.validacao_imagem_id = imagem_id
            st.session_state.validacao_pontos = []
            st.session_state.validacao_ultimo_evento = None

        if "validacao_pontos" not in st.session_state:
            st.session_state.validacao_pontos = []
        if "validacao_ultimo_evento" not in st.session_state:
            st.session_state.validacao_ultimo_evento = None

        st.write("📌 **Instruções de clique:**")
        st.caption(
            "1. Clique na base da sombra.  |  "
            "2. Clique na ponta da sombra."
        )

        img_display, escala_x, escala_y, largura_display = preparar_imagem_cliques(
            img_pil,
            st.session_state.validacao_pontos,
        )

        # O componente retorna x/y do clique na imagem exibida.
        click_data = streamlit_image_coordinates(
            img_display,
            width=largura_display,
            key=f"marcador_validacao_{imagem_id}",
            cursor="crosshair",
        )

        if click_data is not None and len(st.session_state.validacao_pontos) < 2:
            evento_id = click_data.get("unix_time")
            if evento_id is None:
                # Fallback caso alguma versão do componente não retorne unix_time.
                evento_id = (click_data.get("x"), click_data.get("y"))

            if evento_id != st.session_state.validacao_ultimo_evento:
                x_display = click_data.get("x")
                y_display = click_data.get("y")

                if x_display is not None and y_display is not None:
                    x_real = int(round(x_display * escala_x))
                    y_real = int(round(y_display * escala_y))

                    # Garante que o ponto nunca ultrapasse os limites da imagem.
                    x_real = max(0, min(x_real, img_pil.width - 1))
                    y_real = max(0, min(y_real, img_pil.height - 1))

                    st.session_state.validacao_pontos.append((x_real, y_real))
                    st.session_state.validacao_ultimo_evento = evento_id
                    st.rerun()

        pontos_salvos = st.session_state.validacao_pontos

        col_status, col_reset = st.columns([4, 1])
        with col_reset:
            if st.button("↺ Limpar pontos"):
                st.session_state.validacao_pontos = []
                st.session_state.validacao_ultimo_evento = None
                st.rerun()

        with col_status:
            if len(pontos_salvos) == 0:
                st.warning("⚠️ Clique na **base da sombra** para marcar o 1º ponto.")
            elif len(pontos_salvos) == 1:
                x1, y1 = pontos_salvos[0]
                st.warning(
                    f"⚠️ Base marcada em ({x1}, {y1}). "
                    "Agora clique na **ponta da sombra**."
                )
            elif len(pontos_salvos) == 2:
                pts_poste = [tuple(pontos_salvos[0]), tuple(pontos_salvos[1])]
                distancia_sombra_px, azimute_calculado = calcular_dados_referencia(
                    pts_poste
                )

                (x1_real, y1_real), (x2_real, y2_real) = pts_poste
                st.success(
                    f"📍 **Base:** ({x1_real}, {y1_real}) | "
                    f"**Ponta:** ({x2_real}, {y2_real})"
                )
                st.info(
                    f"📏 **Comprimento medido:** {distancia_sombra_px:.1f} pixels | "
                    f"🧭 **Azimute medido:** {azimute_calculado:.1f}°"
                )

    else:
        col_prev, _ = st.columns([1, 2])
        with col_prev:
            st.image(
                img_pil,
                caption="Prévia da Imagem Carregada",
                use_container_width=True,
            )

st.divider()

# -----------------------------------------------------------------------------
# 5. Processamento
# -----------------------------------------------------------------------------
if st.button("🚀 Executar Processamento", type="primary"):

    if imagem_file is None:
        st.warning("⚠️ **Atenção:** Por favor, faça o upload de uma imagem.")
        st.stop()

    if (
        modo_metodo == "Verificação por Referência (Marcar Sombra)"
        and (pts_poste is None or distancia_sombra_px <= 0)
    ):
        st.error(
            "❌ **Atenção:** selecione os dois pontos (base e ponta) na imagem "
            "antes de executar."
        )
        st.stop()

    # Identificação do modelo YOLO
    script_dir = os.path.dirname(os.path.abspath(__file__))
    modelo_nome = "Melhor_modelo_05-12-23.pt"
    caminho_local_script = os.path.join(script_dir, modelo_nome)
    caminho_raiz_projeto = os.path.join(os.getcwd(), modelo_nome)

    path_modelo_usar = None
    if not usar_modelo_padrao:
        if modelo_file is None:
            st.error("❌ Envie o arquivo do modelo `.pt`.")
            st.stop()
        else:
            os.makedirs("temp", exist_ok=True)
            path_modelo_usar = os.path.join("temp", modelo_file.name)
            with open(path_modelo_usar, "wb") as f:
                f.write(modelo_file.getbuffer())
    else:
        if os.path.exists(caminho_local_script):
            path_modelo_usar = caminho_local_script
        elif os.path.exists(caminho_raiz_projeto):
            path_modelo_usar = caminho_raiz_projeto
        else:
            st.error(f"❌ Modelo `{modelo_nome}` não encontrado.")
            st.stop()

    os.makedirs("temp", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    temp_img_path = os.path.join("temp", imagem_file.name)
    with open(temp_img_path, "wb") as f:
        f.write(imagem_file.getbuffer())

    output_path = os.path.join("output", f"Resultado_{imagem_file.name}")

    with st.spinner("⏳ Processando imagem..."):
        try:
            if modo_metodo == "Estimativa Sol/GSD (Geográfica e Solar)":
                hora_local = datetime.datetime.combine(data_captura, hora_captura)
                hora_utc = hora_local - datetime.timedelta(hours=fuso)
                hora_utc = hora_utc.replace(tzinfo=datetime.timezone.utc)

                if modo_escala == "GSD (m/px)" and gsd_direto is not None:
                    gsd_calculado = gsd_direto
                else:
                    gsd_calculado = calcular_gsd_sasplanet(zoom, lat)

                elevacao_solar = get_altitude(lat, long, hora_utc)
                azimute_sol = get_azimuth(lat, long, hora_utc)
                azimute_sombra_calc = (azimute_sol + 180) % 360

                analisar_arvores_sombras(
                    model_path=path_modelo_usar,
                    image_path=temp_img_path,
                    azimute_sombra_graus=azimute_sombra_calc,
                    elevacao_solar_graus=elevacao_solar,
                    gsd=gsd_calculado,
                    output_path=output_path,
                )
            else:
                # No modo de validação, a altura e o azimute são obtidos
                # exclusivamente a partir da referência marcada pelo usuário.
                fator_h = h_ref / distancia_sombra_px

                analisar_arvores_hibrido(
                    model_path=path_modelo_usar,
                    img_original=temp_img_path,
                    azimute_sombra_graus=azimute_calculado,
                    fator_altura_por_pixel=fator_h,
                    gsd_real_mapa=None,
                    pontos_referencia=pts_poste,
                    output_path=output_path,
                )

            # Exibição do Resultado
            if os.path.exists(output_path):
                st.success("✅ **Processamento concluído com sucesso!**")
                st.subheader("📸 Resultado da Análise")
                st.image(
                    output_path,
                    caption="Imagem Processada",
                    use_container_width=True,
                )

                with open(output_path, "rb") as file:
                    st.download_button(
                        label="💾 Baixar Imagem Processada",
                        data=file,
                        file_name=f"Resultado_{imagem_file.name}",
                        mime="image/png",
                    )
            else:
                st.error("❌ O arquivo de imagem gerado não foi localizado.")

        except Exception as err:
            st.error(f"❌ Ocorreu um erro no processamento: `{err}`")