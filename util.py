import numpy as np
import sys

def calc_MI_feat_target(column, target, num_bins):

    def discret(val, bins):
        for idx in xrange(len(bins)):
            try:
                if val >= bins[idx] and val <= bins[idx+1] :
                    return idx
            except:
                print bins
                print val
                sys.exit(-1)


    p_neg = 0.039568534596158902
    p_pos = 0.96043146540384106
    column = column.round(5)
    min_val = np.min(column)
    max_val = np.max(column)
    column.fillna(-999, inplace=True)
    values = column.values
    try:
        bins = [-999999] + np.arange(min_val, max_val, (max_val-min_val)/float(num_bins)).tolist()
        #print '%f - %f' % (min_val, max_val)
        bins[-1] = max_val
        densitys, bin_edges = np.histogram(values, bins=bins, density=True)
    except:
        print bins
        print min_val
        print max_val
        print (max_val-min_val)/num_bins
        sys.exit(-1)
    dist_vals = []
    for val in column.values:
        dist_val = discret(val, bins)
        dist_vals.append(dist_val)

    final_mi = 0
    dist_vals = np.array(dist_vals)
    for level in xrange(len(bins)):
        p_cate_pos = np.sum((dist_vals == level) & (target == 1)) / float(column.shape[0])
        p_cate_neg = np.sum((dist_vals == level) & (target == 0)) / float(column.shape[0])
        p_cate = np.sum((dist_vals == level)) / float(column.shape[0])
        if p_cate_pos == 0 or p_cate_neg == 0:
            continue
        final_mi += p_cate_pos * np.log2(p_cate_pos / (p_cate * p_pos))
        final_mi += p_cate_neg * np.log2(p_cate_neg / (p_cate * p_neg))

    return final_mi

