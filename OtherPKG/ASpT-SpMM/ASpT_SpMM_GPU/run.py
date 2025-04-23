import os

ls = [i for i in os.listdir('/root/SparseCraft/Matrices') if i.endswith(".mtx") and not i.startswith('_') and not 'mawi' in i and not 'kmer' in i]
ls = [i for i in ls if i not in ['2777_kron_g500-logn19.mtx', '2804_ljournal-2008.mtx', '2808_kron_g500-logn20.mtx', '2764_GL7d18.mtx', '2800_com-LiveJournal.mtx', '2785_nv2.mtx', '2769_wikipedia-20060925.mtx', '2846_webbase-2001.mtx', '2770_GL7d19.mtx', '2778_wikipedia-20070206.mtx', '2842_arabic-2005.mtx', '2829_com-Orkut.mtx', '2844_mycielskian19.mtx', '2796_hugebubbles-00020.mtx', '2847_it-2004.mtx', '2799_soc-LiveJournal1.mtx', '2775_wikipedia-20061104.mtx', '2848_twitter7.mtx', '2826_kron_g500-logn21.mtx', '2819_rgg_n_2_23_s0.mtx', '2845_uk-2005.mtx', '2830_rgg_n_2_24_s0.mtx', '2791_hugebubbles-00010.mtx', '2843_nlpkkt240.mtx', '2766_sx-stackoverflow.mtx', '2837_stokes.mtx', '2793_rgg_n_2_22_s0.mtx']]

for mtx in ls:
    print(f'./dspmm_32 /matrix/{mtx} 8')
    os.system(f'./dspmm_32 /matrix/{mtx} 8')
