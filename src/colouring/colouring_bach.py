
import numpy as np
from os.path import join
from colouring import check_or_create, post_process_out
from tqdm import trange
from skimage import io
io.use_plugin('tifffile')
from joblib import Parallel, delayed

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--input", dest="input", type="string",
                      help="record name")
    parser.add_option("--slide", dest="slide", type="string",
                      help="slide_name")
    parser.add_option("-s", "--no_samples",
                  action="store_false", dest="samples", default=True,
                  help="If to save samples")
    parser.add_option("--n_jobs", dest="n_jobs", type="int", default=8,
                      help="Number of jobs")
    (options, _) = parser.parse_args() 

    file = options.input
    tiles_prob = "./tiles_prob"
    tiles_contours = "./tiles_contours"
    tiles_bin = "./tiles_bin"

    folders = [tiles_bin, tiles_contours, tiles_prob]
    out_names = [join(f, f.split('_')[-1] + "_{:03d}.tif") for f in folders]

    for f in folders:
        check_or_create(f)

    files = np.load(file)
    raw = files["raw"]
    segmented_tiles = files["tiles"]
    n = segmented_tiles.shape[0]
    s = segmented_tiles.shape[1]
    bins = np.zeros((n, s, s), dtype="uint8")

    def process_i(i):
        prob = segmented_tiles[i].copy()
        rgb = raw[i]
        list_img = post_process_out(prob, rgb)
        if options.samples:
            for image, name in zip(list_img, out_names):
                io.imsave(name.format(i+1), image, resolution=[1.0, 1.0])
        return list_img[0]


    labeled_bins = Parallel(n_jobs=options.n_jobs)(delayed(process_i)(i) for i in trange(n))
    bins = np.stack(labeled_bins)

    # for i in trange(n):
    #     para = positions[i]
    #     prob = segmented_tiles[i]
    #     rgb = raw[i]
    #     list_img = post_process_out(prob, rgb)
    #     bins[i] = list_img[0]
    #     inp = list(para)
    #     del inp[-2]
    #     if options.samples:
    #         for image, name in zip(list_img, out_names):
    #             io.imsave(name.format(*inp), image, resolution=[1.0, 1.0])
    np.savez("segmented_tiles_and_bins.npz", tiles=segmented_tiles,
            raw=raw, bins=bins)

if __name__ == '__main__':
    main()

