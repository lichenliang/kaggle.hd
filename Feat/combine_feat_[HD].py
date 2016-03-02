
"""
__file__

    combine_feat_[svd100_and_bow_Jun27]_[High].py

__description__

    This file generates one combination of feature set (High).
    Such features are used to generate the best single model with linear model, e.g.,
        - XGBoost linear booster with MSE objective
        - Sklearn Ridge

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys
sys.path.append("../")
from param_config import config
from gen_info import gen_info
from combine_feat import combine_feat, SimpleTransform


if __name__ == "__main__":

    feat_names = [

        ## jaccard coef
        ('jaccard_coef_of_unigram_between_query_title', SimpleTransform()),
        ('jaccard_coef_of_unigram_between_query_description', SimpleTransform()),
        ('jaccard_coef_of_unigram_between_title_description', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_query_title', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_query_description', SimpleTransform()),
        ('jaccard_coef_of_bigram_between_title_description', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_query_title', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_query_description', SimpleTransform()),
        ('jaccard_coef_of_trigram_between_title_description', SimpleTransform()),

        ## dice dist
        ('dice_dist_of_unigram_between_query_title', SimpleTransform()),
        ('dice_dist_of_unigram_between_query_description', SimpleTransform()),
        ('dice_dist_of_unigram_between_title_description', SimpleTransform()),
        ('dice_dist_of_bigram_between_query_title', SimpleTransform()),
        ('dice_dist_of_bigram_between_query_description', SimpleTransform()),
        ('dice_dist_of_bigram_between_title_description', SimpleTransform()),
        ('dice_dist_of_trigram_between_query_title', SimpleTransform()),
        ('dice_dist_of_trigram_between_query_description', SimpleTransform()),
        ('dice_dist_of_trigram_between_title_description', SimpleTransform()),

    ]

    gen_info(feat_path_name="HD_dist")
    combine_feat(feat_names, feat_path_name="HD_dist")