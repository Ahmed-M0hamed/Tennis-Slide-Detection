import cv2 
import numpy as np 


SCALE  = 50      # pixels per metre
H_MARGIN = 300
W_MARGIN =50                  # blank border around court (px)

# Standard ITF dimensions
COURT_W         = 10.97   # doubles width
COURT_H         = 23.77   # full length
SINGLES_W       = 8.23
SERVICE_BOX_H   = 6.40
NET_Y           = COURT_H / 2


def _m2p(x_m, y_m,
         scale = SCALE, h_margin = H_MARGIN , w_margin = W_MARGIN) :
    """Convert metres → pixel (col, row)."""
    return (int(x_m * scale) + w_margin,
            int(y_m * scale) + h_margin)
def build_court_template(scale = SCALE,
                         h_margin = H_MARGIN ,HEAT = False, w_margin = W_MARGIN , BACKGROUND = (204, 37, 58)):


    W_px = int(COURT_W * scale) + 2 * w_margin
    H_px = int(COURT_H * scale) + 2 * h_margin

    img = np.zeros((H_px, W_px, 3), dtype=np.uint8)
    img[:] = BACKGROUND   # ITF green

    def mp(x, y):
        return _m2p(x, y, scale)

    WHITE = (255, 255, 255)
    T = 4

    so = (COURT_W - SINGLES_W) / 2   # singles sideline offset

    # Compute all key pixel coords
    dtl = mp(0,           0)
    dtr = mp(COURT_W,     0)
    dbr = mp(COURT_W,     COURT_H)
    dbl = mp(0,           COURT_H)
    stl = mp(so,          0)
    str_ = mp(so + SINGLES_W, 0)
    sbr = mp(so + SINGLES_W, COURT_H)
    sbl = mp(so,          COURT_H)
    net_l  = mp(0,        NET_Y)
    net_r  = mp(COURT_W,  NET_Y)
    svtl = mp(so,          SERVICE_BOX_H)
    svtr = mp(so + SINGLES_W, SERVICE_BOX_H)
    svbr = mp(so + SINGLES_W, COURT_H - SERVICE_BOX_H)
    svbl = mp(so,          COURT_H - SERVICE_BOX_H)
    svc_mid_t = mp(COURT_W / 2, SERVICE_BOX_H)
    svc_mid_b = mp(COURT_W / 2, COURT_H - SERVICE_BOX_H)

    # Draw lines
    cv2.rectangle(img, dtl, dbr, WHITE, T)          # doubles outline
    cv2.line(img, stl,  sbl,  WHITE, T)             # singles left
    cv2.line(img, str_, sbr,  WHITE, T)             # singles right
    cv2.line(img, net_l, net_r, WHITE, T + 1)       # net (thicker)
    cv2.line(img, svtl, svtr, WHITE, T)             # top service line
    cv2.line(img, svbl, svbr, WHITE, T)             # bottom service line
    cv2.line(img, svc_mid_t, svc_mid_b, WHITE, T)  # centre service line

    # Center marks on baselines
    cx = int(COURT_W / 2 * scale) + w_margin
    cv2.line(img, (cx, w_margin), (cx, w_margin + 8), WHITE, T)
    cv2.line(img, (cx, H_px - w_margin - 8), (cx, H_px - w_margin), WHITE, T)

    kp_world = np.array([
        dtl,   # 0
        dtr,   # 1
        dbr,   # 2
        dbl,   # 3
        stl,   # 4
        str_,  # 5
        sbr,   # 6
        sbl,   # 7
        svtl ,
        svtr , 
        svbr , 
        svbl  , 
        svc_mid_t , 
        svc_mid_b , 



    ], dtype=np.float32)
    
    return img , kp_world 