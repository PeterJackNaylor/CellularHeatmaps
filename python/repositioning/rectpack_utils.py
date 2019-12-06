
import skimage as sk 
import numpy as np

from rectpack import newPacker

## not commented
def get_rectangles(labeled):
    rect = []
    dic = {}
    cc = sk.measure.regionprops(labeled)
    for cc_i in sk.measure.regionprops(labeled):
        x, y, w, h = cc_i.bbox
        #removing object around
        dic[cc_i.label] = [x, y, w, h], cc_i.image
        rect.append([w - x, h - y, cc_i.label])
    return rect, dic

  
def move_rectangles(packer, image, dic, b):
    z = image.shape[2]
    res = np.zeros(shape=(b[0], b[1], z))
    abin = packer[0]
    for rect in abin:
        x_top_l, y_top_l = rect.corner_top_l.x, rect.corner_top_l.y
        x_bot_r, y_bot_r = rect.corner_bot_r.x, rect.corner_bot_r.y
        rid = rect.rid
        sizes, binary = dic[rid]
        x, y, w, h = sizes 
        try:
            img_tmp = image.copy()[x:w, y:h]
            img_tmp[(1 - binary.astype(int)).astype(bool)] = 0
            res[x_top_l:x_bot_r, y_bot_r:y_top_l] = img_tmp 
        except:
            img_tt = image.copy()[x:w, y:h].transpose((1, 0, 2))
            img_tt[(1 - binary.astype(int)).astype(bool).transpose((1, 0))] = 0
            res[x_top_l:x_bot_r, y_bot_r:y_top_l] = img_tt 
    return res

def place_rectangles(labeled, image):
    
    rectangles, mapping = get_rectangles(labeled)
    Mbins = [(112 + x, 112 + x) for x in range(0, 1000, 10)]
    
    for b in Mbins:
        packer = newPacker()
        for r in rectangles:
            packer.add_rect(*r)
        packer.add_bin(*b)
        packer.pack()
        try:
            if len(packer[0]) == labeled.max():
                break
        except IndexError:
            pass
    res = move_rectangles(packer, image, mapping, b)
    return res, mapping

