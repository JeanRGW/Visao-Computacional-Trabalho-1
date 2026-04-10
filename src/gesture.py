import cv2
import numpy as np
import time
import collections

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE    = 0.0
    PYAUTOGUI_OK = True
except ImportError:
    PYAUTOGUI_OK = False


class Config:
    CAM_W, CAM_H, CAM_FPS = 640, 480, 30

    HSV_BAIXO_1 = np.array([0,  20,  60], dtype=np.uint8)
    HSV_ALTO_1  = np.array([25, 255, 255], dtype=np.uint8)
    HSV_BAIXO_2 = np.array([160, 20, 60],  dtype=np.uint8)
    HSV_ALTO_2  = np.array([180, 255, 255], dtype=np.uint8)

    YCRCB_BAIXO = np.array([0,  133, 77],  dtype=np.uint8)
    YCRCB_ALTO  = np.array([255, 173, 127], dtype=np.uint8)

    MORPH_KERNEL    = 5
    AREA_MIN_MAO    = 3000

    LK_WIN_SIZE  = (21, 21)
    LK_MAX_LEVEL = 3
    LK_CRITERIA  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    ST_MAX_CORNERS = 80
    ST_QUALITY     = 0.25
    ST_MIN_DIST    = 8

    LIMIAR_DESLOCAMENTO = 80
    LIMIAR_VERT_MAX     = 50
    COOLDOWN            = 0.9
    JANELA_FRAMES       = 14
    MIN_PONTOS          = 6

    FREQ_REDETECCAO = 8


class SegmentadorMao:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.MORPH_KERNEL, cfg.MORPH_KERNEL)
        )
        self.hsv_baixo_1 = cfg.HSV_BAIXO_1.copy()
        self.hsv_alto_1  = cfg.HSV_ALTO_1.copy()
        self.hsv_baixo_2 = cfg.HSV_BAIXO_2.copy()
        self.hsv_alto_2  = cfg.HSV_ALTO_2.copy()
        self.ycrcb_baixo = cfg.YCRCB_BAIXO.copy()
        self.ycrcb_alto  = cfg.YCRCB_ALTO.copy()

    def segmentar(self, frame):
        hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1       = cv2.inRange(hsv, self.hsv_baixo_1, self.hsv_alto_1)
        m2       = cv2.inRange(hsv, self.hsv_baixo_2, self.hsv_alto_2)
        mask_hsv = cv2.bitwise_or(m1, m2)

        ycrcb    = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycc = cv2.inRange(ycrcb, self.ycrcb_baixo, self.ycrcb_alto)

        mascara = cv2.bitwise_and(mask_hsv, mask_ycc)
        mascara = cv2.erode(mascara,  self.kernel, iterations=1)
        mascara = cv2.dilate(mascara, self.kernel, iterations=3)
        mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
        _, mascara = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return mascara, None, None

        maior = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(maior) < self.cfg.AREA_MIN_MAO:
            return mascara, None, None

        mascara_limpa = np.zeros_like(mascara)
        cv2.drawContours(mascara_limpa, [maior], -1, 255, -1)

        bbox = cv2.boundingRect(maior)

        M = cv2.moments(maior)
        centroide = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] > 0 else None

        return mascara_limpa, bbox, centroide

    def calibrar(self, frame, bbox):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        pad = min(w, h) // 4
        roi = frame[max(0, cy-pad):cy+pad, max(0, cx-pad):cx+pad]

        if roi.size == 0:
            return

        hsv_roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        h_mean, s_mean = float(np.mean(hsv_roi[:,:,0])), float(np.mean(hsv_roi[:,:,1]))
        h_std,  s_std  = float(np.std(hsv_roi[:,:,0])),  float(np.std(hsv_roi[:,:,1]))

        self.hsv_baixo_1 = np.array([max(0,   int(h_mean - 2*h_std - 5)),
                                     max(15,  int(s_mean - 2*s_std)), 50], dtype=np.uint8)
        self.hsv_alto_1  = np.array([min(180, int(h_mean + 2*h_std + 5)), 255, 255], dtype=np.uint8)

        cr_mean = float(np.mean(ycrcb_roi[:,:,1]))
        cb_mean = float(np.mean(ycrcb_roi[:,:,2]))
        cr_std  = float(np.std(ycrcb_roi[:,:,1]))
        cb_std  = float(np.std(ycrcb_roi[:,:,2]))

        self.ycrcb_baixo = np.array([0,
                                     max(100, int(cr_mean - 2*cr_std - 8)),
                                     max(60,  int(cb_mean - 2*cb_std - 8))], dtype=np.uint8)
        self.ycrcb_alto  = np.array([255,
                                     min(200, int(cr_mean + 2*cr_std + 8)),
                                     min(160, int(cb_mean + 2*cb_std + 8))], dtype=np.uint8)

        print(f"  [CAL] HSV: {self.hsv_baixo_1} – {self.hsv_alto_1}")
        print(f"  [CAL] YCrCb: {self.ycrcb_baixo} – {self.ycrcb_alto}")


class RastreadorLK:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lk_params = dict(
            winSize  = cfg.LK_WIN_SIZE,
            maxLevel = cfg.LK_MAX_LEVEL,
            criteria = cfg.LK_CRITERIA,
        )
        self.st_params = dict(
            maxCorners   = cfg.ST_MAX_CORNERS,
            qualityLevel = cfg.ST_QUALITY,
            minDistance  = cfg.ST_MIN_DIST,
            blockSize    = 7,
        )
        self.cinza_ant  = None
        self.pontos_ant = None

    def detectar_pontos(self, cinza, mascara):
        pts = cv2.goodFeaturesToTrack(cinza, mask=mascara, **self.st_params)
        if pts is not None and len(pts) > 0:
            self.cinza_ant  = cinza.copy()
            self.pontos_ant = pts
            return True
        return False

    def rastrear(self, cinza_atual):
        if self.cinza_ant is None or self.pontos_ant is None:
            return None, None, 0

        pts_novos, status, erro = cv2.calcOpticalFlowPyrLK(
            self.cinza_ant, cinza_atual, self.pontos_ant, None, **self.lk_params
        )

        if pts_novos is None or status is None:
            return None, None, 0

        mascara_ok = status.flatten() == 1

        if erro is not None:
            ef  = erro.flatten()
            med = np.median(ef[mascara_ok]) if mascara_ok.sum() > 0 else 999
            mascara_ok &= (ef < max(med * 2.5, 8.0))

        n = int(mascara_ok.sum())
        if n < self.cfg.MIN_PONTOS:
            return None, None, n

        bons_novos = pts_novos[mascara_ok].reshape(-1, 2)
        bons_ant   = self.pontos_ant[mascara_ok].reshape(-1, 2)

        self.cinza_ant  = cinza_atual.copy()
        self.pontos_ant = bons_novos.reshape(-1, 1, 2)

        return bons_novos, bons_ant, n

    def resetar(self):
        self.cinza_ant  = None
        self.pontos_ant = None


class AnalisadorGesto:
    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.hist_dx      = collections.deque(maxlen=cfg.JANELA_FRAMES)
        self.hist_dy      = collections.deque(maxlen=cfg.JANELA_FRAMES)
        self._ultimo_t    = 0.0
        self.ultimo_gesto = "Nenhum"
        self.fator_sensib = 1.0

    @property
    def limiar(self):
        return self.cfg.LIMIAR_DESLOCAMENTO * self.fator_sensib

    def atualizar(self, dx, dy):
        self.hist_dx.append(float(dx))
        self.hist_dy.append(float(dy))

    def verificar(self):
        if len(self.hist_dx) < 4:
            return None
        if (time.time() - self._ultimo_t) < self.cfg.COOLDOWN:
            return None

        dx_arr = np.array(self.hist_dx)
        dy_arr = np.array(self.hist_dy)

        dx_acum      = float(dx_arr.sum())
        dy_acum      = float(np.abs(dy_arr).sum())
        consistencia = (dx_arr > 0).mean() if dx_acum > 0 else (dx_arr < 0).mean()
        dx_pico      = float(np.abs(dx_arr).max())

        if (abs(dx_acum)  > self.limiar
                and dy_acum      < self.cfg.LIMIAR_VERT_MAX
                and consistencia > 0.60
                and dx_pico      > 4.0):

            self._ultimo_t = time.time()
            self.hist_dx.clear()
            self.hist_dy.clear()

            if dx_acum > 0:
                self.ultimo_gesto = "DIREITA ->"
                return "direita"
            else:
                self.ultimo_gesto = "<- ESQUERDA"
                return "esquerda"

        return None

    def ajustar_sensibilidade(self, delta):
        self.fator_sensib = max(0.3, min(2.5, self.fator_sensib + delta))

    def progresso(self):
        if not self.hist_dx:
            return 0.0, 0
        acum = sum(self.hist_dx)
        return min(abs(acum) / self.limiar, 1.0), (1 if acum > 0 else -1)


class HUD:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.trilha_centroide = collections.deque(maxlen=25)

    def _t(self, frame, txt, pos, escala, cor, esp=1):
        cv2.putText(frame, txt, pos, self.FONT, escala, (0, 0, 0), esp + 2, cv2.LINE_AA)
        cv2.putText(frame, txt, pos, self.FONT, escala, cor,        esp,     cv2.LINE_AA)

    def _painel(self, frame, x1, y1, x2, y2, alpha=0.58):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), (12, 12, 12), -1)
        cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

    def desenhar(self, frame, estado, analisador, mascara, bbox, centroide,
                 bons_novos, bons_ant, debug):
        h, w = frame.shape[:2]

        if mascara is not None:
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                cv2.drawContours(frame, [max(contornos, key=cv2.contourArea)],
                                 -1, (0, 220, 100), 2, cv2.LINE_AA)

        if bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 180, 255), 1)

        if centroide is not None:
            self.trilha_centroide.append(centroide)
            for i in range(1, len(self.trilha_centroide)):
                a = float(i) / len(self.trilha_centroide)
                cv2.line(frame, self.trilha_centroide[i-1], self.trilha_centroide[i],
                         (int(0*a), int(220*a), int(100*a)), 2, cv2.LINE_AA)
            cv2.circle(frame, centroide, 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, centroide, 4, (0, 180, 255), -1, cv2.LINE_AA)

        if bons_novos is not None and bons_ant is not None:
            for novo, ant in zip(bons_novos, bons_ant):
                cv2.line(frame,  tuple(ant.astype(int)), tuple(novo.astype(int)), (80, 255, 180), 1, cv2.LINE_AA)
                cv2.circle(frame, tuple(novo.astype(int)), 2, (0, 120, 255), -1, cv2.LINE_AA)

        prog, direcao = analisador.progresso()
        if prog > 0.2 and centroide is not None:
            comp = int(55 * prog)
            cx, cy_c = centroide
            cv2.arrowedLine(frame,
                            (cx - direcao * comp, cy_c),
                            (cx + direcao * comp, cy_c),
                            (80, 255, 160) if direcao > 0 else (255, 160, 80),
                            3, cv2.LINE_AA, tipLength=0.4)

        agora = time.time()
        if (agora - estado.get("t_ultimo_gesto", 0)) < 0.45:
            gt = estado.get("ultimo_gesto_tipo", "")
            ov = frame.copy()
            if gt == "direita":
                cv2.rectangle(ov, (w-90, 0), (w, h), (0, 255, 120), -1)
                cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
                self._t(frame, ">>>", (w-80, h//2), 1.4, (0, 255, 120), 2)
            elif gt == "esquerda":
                cv2.rectangle(ov, (0, 0), (90, h), (255, 140, 0), -1)
                cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
                self._t(frame, "<<<", (10, h//2), 1.4, (255, 140, 0), 2)

        self._painel(frame, 5, 5, 318, 182)
        self._t(frame, "Interface Gestual - Slides", (12, 26), 0.54, (255, 230, 80), 1)

        cor_st = (80, 230, 80) if estado["mao_detectada"] else (80, 160, 255)
        self._t(frame, f"Status : {estado['status']}",                         (12, 50),  0.48, cor_st)
        self._t(frame, f"Pontos : {estado['n_pontos']}     Sensib: {analisador.fator_sensib:.1f}x",
                                                                                (12, 72),  0.46, (200, 200, 200))
        self._t(frame, f"Gesto  : {analisador.ultimo_gesto}",                  (12, 94),  0.48, (200, 200, 255))
        self._t(frame, "Q=sair R=reset D=debug C=calibrar +/-=sensib",         (12, 116), 0.37, (130, 130, 130))

        bx, by_, bw2, bh2 = 12, 132, 288, 13
        cv2.rectangle(frame, (bx, by_), (bx+bw2, by_+bh2), (50, 50, 50), -1)
        fill   = int(prog * bw2)
        cor_b  = (80, 255, 160) if direcao >= 0 else (255, 160, 80)
        if fill > 0:
            if direcao >= 0:
                cv2.rectangle(frame, (bx, by_), (bx+fill, by_+bh2), cor_b, -1)
            else:
                cv2.rectangle(frame, (bx+bw2-fill, by_), (bx+bw2, by_+bh2), cor_b, -1)
        cv2.rectangle(frame, (bx, by_), (bx+bw2, by_+bh2), (110, 110, 110), 1)
        mx = bx + bw2 // 2
        cv2.line(frame, (mx, by_-2), (mx, by_+bh2+2), (200, 200, 200), 1)
        self._t(frame, "Progresso:", (12, 160), 0.38, (150, 150, 150))

        if debug:
            if mascara is not None:
                mini     = cv2.resize(mascara, (120, 90))
                mini_rgb = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
                frame[h-95:h-5, w-125:w-5] = mini_rgb
                self._t(frame, "mascara", (w-122, h-8), 0.35, (200, 200, 100))
            if bons_novos is not None:
                for i, pt in enumerate(bons_novos):
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
                    self._t(frame, str(i), (px+4, py-3), 0.28, (255, 255, 0))

        return frame


def executar_interface_gestual(camera_id: int = 0):
    print("\n" + "="*55)
    print("  INTERFACE GESTUAL – CONTROLE DE SLIDES")
    print("  Metodo: Segmentacao de pele (HSV+YCrCb) + Lucas-Kanade")
    print("="*55)
    print("  -> Mao para DIREITA  : proximo slide")
    print("  <- Mao para ESQUERDA : slide anterior")
    print("  [Q] Sair  [R] Reset  [D] Debug  [C] Calibrar  [+/-] Sensib")
    print("="*55 + "\n")

    if not PYAUTOGUI_OK:
        print("  [AVISO] pyautogui nao instalado – gestos detectados mas teclas NAO serao enviadas.\n")

    cfg         = Config()
    segmentador = SegmentadorMao(cfg)
    rastreador  = RastreadorLK(cfg)
    analisador  = AnalisadorGesto(cfg)
    hud         = HUD()

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"  [ERRO] Camera {camera_id} nao acessivel.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAM_H)
    cap.set(cv2.CAP_PROP_FPS,          cfg.CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    estado = {
        "status":            "Aguardando mao...",
        "mao_detectada":     False,
        "n_pontos":          0,
        "ultimo_gesto_tipo": "",
        "t_ultimo_gesto":    0.0,
    }

    debug       = False
    frame_count = 0

    print("  Camera aberta. Coloque a mao em frente a camera.\n")
    print("  DICA: Use fundo escuro para melhor deteccao.")
    print("  DICA: Pressione [C] com a mao visivel para calibrar a cor.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame       = cv2.flip(frame, 1)
        cinza_atual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1

        mascara, bbox, centroide = segmentador.segmentar(frame)
        mao_presente = bbox is not None

        if mao_presente:
            estado["mao_detectada"] = True
            estado["status"]        = "Mao detectada"

            if frame_count % cfg.FREQ_REDETECCAO == 0 or rastreador.pontos_ant is None:
                if not rastreador.detectar_pontos(cinza_atual, mascara):
                    estado["status"] = "Sem pontos na mao"

            bons_novos, bons_ant, n = rastreador.rastrear(cinza_atual)
            estado["n_pontos"] = n

            if bons_novos is not None and n >= cfg.MIN_PONTOS:
                estado["status"] = f"Rastreando {n} pontos"
                d  = bons_novos - bons_ant
                dx = float(np.median(d[:, 0]))
                dy = float(np.median(d[:, 1]))
                analisador.atualizar(dx, dy)
            else:
                if len(hud.trilha_centroide) >= 2 and centroide is not None:
                    prev = hud.trilha_centroide[-1]
                    analisador.atualizar(centroide[0] - prev[0], centroide[1] - prev[1])
        else:
            estado["mao_detectada"] = False
            estado["status"]        = "Aguardando mao..."
            estado["n_pontos"]      = 0
            rastreador.resetar()
            bons_novos, bons_ant = None, None

        gesto = analisador.verificar()
        if gesto:
            estado["ultimo_gesto_tipo"] = gesto
            estado["t_ultimo_gesto"]    = time.time()
            if gesto == "direita":
                print("  -> GESTO DIREITA -> SETA DIREITA")
                if PYAUTOGUI_OK:
                    pyautogui.press("right")
            else:
                print("  <- GESTO ESQUERDA -> SETA ESQUERDA")
                if PYAUTOGUI_OK:
                    pyautogui.press("left")

        frame = hud.desenhar(frame, estado, analisador,
                             mascara, bbox, centroide,
                             bons_novos, bons_ant, debug)
        cv2.imshow("Interface Gestual - Slides (Q para sair)", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord('q'), ord('Q'), 27):
            print("  Encerrando...")
            break
        elif tecla in (ord('r'), ord('R')):
            rastreador.resetar()
            analisador.hist_dx.clear()
            analisador.hist_dy.clear()
            hud.trilha_centroide.clear()
            print("  Rastreamento reiniciado.")
        elif tecla in (ord('d'), ord('D')):
            debug = not debug
            print(f"  Debug {'ativado' if debug else 'desativado'}.")
        elif tecla in (ord('c'), ord('C')):
            if bbox is not None:
                segmentador.calibrar(frame, bbox)
                print("  Calibracao concluida.")
            else:
                print("  [AVISO] Nenhuma mao detectada para calibrar.")
        elif tecla == ord('+'):
            analisador.ajustar_sensibilidade(-0.1)
            print(f"  Sensibilidade: {analisador.fator_sensib:.1f}x")
        elif tecla == ord('-'):
            analisador.ajustar_sensibilidade(+0.1)
            print(f"  Sensibilidade: {analisador.fator_sensib:.1f}x")

    cap.release()
    cv2.destroyAllWindows()
    print("  Interface gestual encerrada.\n")


class GestureSlideController:
    """Compatibility wrapper for current app integration."""

    def __init__(self, cam_index: int = 0):
        self.cam_index = cam_index

    def run(self):
        executar_interface_gestual(self.cam_index)


if __name__ == "__main__":
    controller = GestureSlideController()
    controller.run()