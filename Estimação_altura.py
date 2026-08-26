import cv2
import numpy as np
from ultralytics import YOLO
import math
import datetime
from pysolar.solar import get_altitude, get_azimuth
import os

# --- FUNÇÕES AUXILIARES ---
def dms_para_decimal(graus, minutos, segundos, direcao): # Conversão de graus para decimais
    decimal = graus + (minutos / 60) + (segundos / 3600)
    if direcao.upper() in ['S', 'W']:
        decimal = -decimal
    return decimal

def calcular_gsd_sasplanet(zoom_level, latitude): # Calcula o tamanha de metro/pixel da imagem
    equator_resolution = 156543.03392
    lat_rad = math.radians(latitude)
    gsd = (equator_resolution * math.cos(lat_rad)) / (2 ** zoom_level)
    return gsd

# --- FUNÇÃO PRINCIPAL ---
def analisar_arvores_sombras(
    model_path, 
    image_path, 
    azimute_sombra_graus, 
    elevacao_solar_graus, 
    gsd,                  
    output_path="resultado_analise.jpg"
):
    print(f"Carregando modelo: {model_path}...") # Carrega o modelo da YOLO
    model = YOLO(model_path)
    
    img = cv2.imread(image_path) # Lê a imagem, verificando sua existncia
    if img is None:
        print("Erro ao abrir imagem.")
        return
    
    h, w = img.shape[:2] # Altura e largura da imagem para assim encontrar o centro
    centro_imagem = (w // 2, h // 2)
    
    vis_img = img.copy() # Cópia da imagem original para desenhar por cima
    cv2.drawMarker(vis_img, centro_imagem, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # Adiciona legenda do Azimute na imagem para faciliatar conferência
    cv2.putText(vis_img, f"Azimute Busca: {azimute_sombra_graus:.1f}", (10, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Inferência e detecção da IA
    print("Rodando detecção...")

    # Melhora o contraste da imagem
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_melhorada = cv2.merge((cl,a,b))
    img_melhorada = cv2.cvtColor(img_melhorada, cv2.COLOR_LAB2BGR)
    

    results = model.predict(img_melhorada, conf=0.25, imgsz=1024) 
    result = results[0]
    
    if result.masks is None: # Caso não encontre árvores
        print("Nenhuma segmentação encontrada!")
        return

    CLASS_ID_SOMBRA = 0
    CLASS_ID_ARVORE = 1
    
    arvores = []
    sombras = []
    
    # Extrai os polígonos (máscaras) e as classes detectadas pela IA
    masks = result.masks.xy 
    classes = result.boxes.cls.cpu().numpy()
    
    # Separação e processamento das detecções
    for i, poly in enumerate(masks):
        if len(poly) == 0: continue
        poly_int = np.array(poly, dtype=np.int32)
        cls_id = int(classes[i])
        
        obj = {'id': i, 'poly': poly_int, 'mask_img': None}
        
        if cls_id == CLASS_ID_ARVORE: # Se for árvore
            # Encontra o menor círculo que cobre a árvore para achar poss´velmente o seu tronco (centro)
            (x, y), raio = cv2.minEnclosingCircle(poly_int)
            obj['center'] = (int(x), int(y))
            obj['radius'] = raio
            arvores.append(obj)
        elif cls_id == CLASS_ID_SOMBRA: # Se for sombra
            sombras.append(obj)

    print(f"Detectados: {len(arvores)} árvores e {len(sombras)} sombras.")

    # Vetor de direção solar transformado:
    # Transforma o azimute em um vetor 2D (x,y) indicando para onde as sombras devem cair
    rad = math.radians(azimute_sombra_graus)
    vec_x = math.sin(rad)
    vec_y = -math.cos(rad) 
    
    raio_busca = 2000 # Tamanho máximo da linha virtual em pixels

    candidatos = [] 

    # Pré-renderiza máscaras das sombras para facilitar análise
    for sombra in sombras:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [sombra['poly']], 255)
        sombra['mask_img'] = m

    # Associar a árvore com a sua possível sombra
    for i_arv, arv in enumerate(arvores):
        cx, cy = arv['center']
        raio = arv['radius']

        # Desenhar a copa da árvore
        cv2.circle(vis_img, (cx, cy), int(raio), (0, 255, 255), 2)
        
        ponta_linha = (int(cx + vec_x * raio_busca), int(cy + vec_y * raio_busca)) # Ponto final da nossa linha virtual de busca
        
        # Cria uma máscara só com a linha virtual traçada
        mask_linha = np.zeros((h, w), dtype=np.uint8)
        thickness = 2 # Fixo e fino para garantir precisão direcional
        cv2.line(mask_linha, (cx, cy), ponta_linha, 255, thickness=thickness)

        for i_somb, sombra in enumerate(sombras):
            mask_sombra = sombra['mask_img']
            intersecao = cv2.bitwise_and(mask_linha, mask_sombra)
            
            if cv2.countNonZero(intersecao) > 0: # Se cruzou, extrai todas as coordenadas dessa interseção
                pontos = cv2.findNonZero(intersecao)
                pontos = pontos.reshape(-1, 2)
                
                # Calcula a distância de todos os pontos de interseção até o tronco
                dists = np.linalg.norm(pontos - np.array([cx, cy]), axis=1)
                
                # Pega a menor distância (onde a sombra começa) e a maior (a ponta final da sombra)
                min_dist = np.min(dists) 
                max_dist = np.max(dists)
                idx_max = np.argmax(dists)
                ponto_final = pontos[idx_max]

                candidatos.append({
                    'dist_inicio': min_dist,
                    'dist_final': max_dist,
                    'arvore_idx': i_arv,
                    'sombra_idx': i_somb,
                    'ponto_final': ponto_final
                })

    # Resolver Conflitos
    candidatos.sort(key=lambda x: x['dist_inicio']) # Ordena para processar primeiro as sombras que estão mais coladas nas árvores

    arvores_processadas = set()
    sombras_processadas = set()
    resultados_finais = []

    for c in candidatos:
        # Faz as separações baseadas nas menores distâncias e associa árvore a apenas uma sombra
        aid = c['arvore_idx']
        sid = c['sombra_idx']

        # Garante que cada árvore só tenha uma sombra, e cada sombra pertença a só uma árvore
        if aid in arvores_processadas or sid in sombras_processadas: 
            continue
        
        arvores_processadas.add(aid)
        sombras_processadas.add(sid)
        resultados_finais.append(c)

    # Desenhar Resultados
    tan_elevacao = math.tan(math.radians(elevacao_solar_graus))

    for res in resultados_finais:
        arv = arvores[res['arvore_idx']]
        cx, cy = arv['center']
        px, py = tuple(res['ponto_final'])
        
        # Conversão de pixels para Metros usando o GSD
        valor_raio_m = arv['radius'] * gsd
        valor_sombra_m = res['dist_final'] * gsd

        # Aplica a trigonometria
        altura_estimada_m = valor_sombra_m * tan_elevacao

        # Linha Vermelha (Sombra Confirmada)
        cv2.line(vis_img, (cx, cy), (px, py), (0, 0, 255), 2)
        cv2.circle(vis_img, (px, py), 4, (0, 0, 255), -1)

        # Monta o texto de resultado para anexar na imagem
        texto = f"R:{valor_raio_m:.1f}m | S:{valor_sombra_m:.1f}m | H:{altura_estimada_m:.1f}m"
        text_y = cy - 10 if cy - 10 > 10 else cy + 20
        (w_text, h_text), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis_img, (cx - 20, text_y - h_text - 2), (cx - 20 + w_text, text_y + 2), (0,0,0), -1)
        cv2.putText(vis_img, texto, (cx - 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Salva imagem processada
    cv2.imwrite(output_path, vis_img)
    print(f"Imagem salva em: {output_path}")


# ---------------------------------------------------------------------- BLOCO DE EXECUÇÃO PRINCIPAL ----------------------------------------------------------------------
if __name__ == "__main__":

    # Configurações de localização e escala -----------------------------------------
    lat_dec = -15.7599053 
    long_dec = -47.8713185 
    zoom_usado = 22 

    # Configurações de tempo --------------------------------------------------------
    hora_local = datetime.datetime(2026, 4, 27, 8, 30, 0)     
    fuso_horario = 3 
    hora_utc = hora_local.replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(hours=fuso_horario) 

    # Cálculos solares e de sombra -------------------------------------------------
    gsd_calculado = calcular_gsd_sasplanet(zoom_usado, lat_dec) 
    elevacao_solar = get_altitude(lat_dec, long_dec, hora_utc) 
    azimute_sol = get_azimuth(lat_dec, long_dec, hora_utc) 
    azimute_sombra_calc = (azimute_sol + 180) % 360   

    print(f"TESTE COM HORA: {hora_local.strftime('%H:%M')}")
    print(f"Sombra deve apontar para Azimute: {azimute_sombra_calc:.2f}°")

    diretorio_script = os.path.dirname(os.path.abspath(__file__))

    # Configurações de Input -------------------------------------------------------
    MODEL_PATH = os.path.abspath(os.path.join(diretorio_script, "Melhor_modelo_05-12-23.pt")) 
    IMAGE_PATH = os.path.abspath(os.path.join(diretorio_script, "input", "Imagem_de_teste2.jpg"))

    # Configurações de Output ------------------------------------------------------
    pasta_saida = os.path.abspath(os.path.join(diretorio_script, "output"))
    if not os.path.exists(pasta_saida): 
        os.makedirs(pasta_saida)

    nome_imagem_original = os.path.basename(IMAGE_PATH) 
    nome_novo_arquivo = f"Estimativa_{nome_imagem_original}" 
    output_path = os.path.join(pasta_saida, nome_novo_arquivo) 

    # Execução da função principal ------------------------------------------------
    analisar_arvores_sombras(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        azimute_sombra_graus=azimute_sombra_calc, 
        elevacao_solar_graus=elevacao_solar,      
        gsd=gsd_calculado,                        
        output_path=output_path
    )